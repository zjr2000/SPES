

<div align="center">

# ⚡ SPES: Sincronización de Expertos Dispersos
### Preentrenamiento de un Modelo de Lenguaje Grande usando GPUs Distribuidas: Un Paradigma Descentralizado Eficiente en Memoria

**Jinrui Zhang**$^{1,2}$, **Chaodong Xiao**$^{1,2}$, **Aoqi Wu**$^{1,2}$, **Xindong Zhang**$^2$, **Lei Zhang**$^{1,2}$

<sup>1</sup>Departamento de Computación, The Hong Kong Polytechnic University  
<sup>2</sup>Instituto de Investigación OPPO

📧 [jin-rui.zhang@connect.polyu.hk](mailto:jin-rui.zhang@connect.polyu.hk)

<br>

[![GitHub](https://img.shields.io/badge/GitHub-SPES-blue?logo=github)](https://github.com/zjr2000/SPES)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2602.11543)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Weights-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/collections/zjr2000/spes)

</div>

---

## 📖 Introducción

**SPES** (**SP**arse **E**xpert **S**ync) es un framework de entrenamiento descentralizado de vanguardia y eficiente en memoria, diseñado para el preentrenamiento de LLMs MoE a través de nodos de GPU distribuidos geográficamente.

A diferencia de los paradigmas convencionales que requieren interconexiones de alto ancho de banda, SPES permite el preentrenamiento colaborativo de modelos **Mixture-of-Experts** (Mezcla de Expertos) donde los nodos operan de manera semi-independiente.

### 🌟 Características Principales

| Característica | Descripción |
| :--- | :--- |
| 🌐 **Entrenamiento Descentralizado** | Opera sin interconexiones inter-nodo de alta velocidad. Cada nodo funciona como una unidad de entrenamiento independiente con DDP local. |
| 💾 **Eficiencia en Memoria** | Los nodos solo mantienen gradientes/estados del optimizador para su subconjunto *local* de expertos, reduciendo drásticamente la huella de memoria. |
| ⚡ **Sincronización Dispersa** | Utiliza un servidor de parámetros gRPC ligero para sincronizar *solo* los parámetros entrenados periódicamente. |
| 🔀 **Fusión Inteligente** | Implementa una fusión ponderada inteligente con un programa de decaimiento alpha para garantizar una convergencia estable durante la transferencia de conocimiento. |

---

## 🚧 Hoja de Ruta y Estado

- [x] **Liberación del Código de Entrenamiento**
- [x] Liberación de puntos de control preentrenados
- [ ] Liberación de registros de entrenamiento

---

## 🤗 Puntos de Control del Modelo

Los puntos de control preentrenados están disponibles en la [colección de SPES en Hugging Face](https://huggingface.co/collections/zjr2000/spes).

| Modelo | Descripción | Puntos de Control | Registro |
| :--- | :--- | :--- | :--- |
| `SPES-2B` | Modelo de 2B entrenado desde cero. | [🤗 Hugging Face](https://huggingface.co/zjr2000/SPES-2B) |  |
| `SPES-7B` | Modelo de 7B entrenado desde cero. | [🤗 Hugging Face](https://huggingface.co/zjr2000/SPES-7B) |  |
| `SPES-9B` | Modelo de 9B inicializado a partir de Qwen3-1.7B. | [🤗 Hugging Face](https://huggingface.co/zjr2000/SPES-9B) | [📈 Registro de Weights & Biases](https://wandb.ai/zjr2000/spes/reports/SPES-9B-Train-Log--VmlldzoxNjI0MzA2Ng?accessToken=ghf43wkxavw7qnoolb9kcaeji2y8yg2dunvzowdid7jn02set7c10e1vc0t1bzi9) |
---

## 🔧 Instalación

### Requisitos Previos
*   **Python:** `>= 3.10`
*   **CUDA:** `>= 12.1` (Tested on 12.4)
*   **PyTorch:** `2.5.1`
*   **Hardware:** NVIDIA GPUs (Tested on A100/A800/L40S)

### Instalación Rápida

```bash
# 1. Clone the repository
git clone https://github.com/zjr2000/SPES.git
cd SPES

# 2. Install PyTorch (Adjust CUDA version if necessary)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install SPES and core dependencies
pip install -e '.[all]'

# 4. Install gRPC components
pip install grpcio==1.73.1 grpcio-tools==1.73.1 protobuf==6.31.0
```

### Dependencias para Evaluación
Para ejecutar benchmarks usando el [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness):
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[hf]"
```

---

## 📦 Preparación de Datos

SPES utiliza archivos numpy memmap tokenizados (`.npy`) para la carga de datos de alto rendimiento.

### 1. Tokenización de Datos Crudos
Convierte tus archivos `.jsonl` o `.parquet` utilizando el script proporcionado:

```bash
python data_process_scripts/tokenize_data.py \
    --file_glob "/path/to/your/data/*.jsonl" \
    --tokenizer_name_or_path "Qwen/Qwen2.5-0.5B" \
    --output_prefix "/path/to/output/tokenized_" \
    --text_field "text" \
    --processes 8 \
    --batch_size 500 \
    --max_shard_bytes 4294967296 \
    --dtype "uint32"
```

### 2. Generación de Lista de Archivos
Crea un archivo de manifiesto para la configuración de entrenamiento:
```bash
bash data_process_scripts/list_processed_files.sh /path/to/tokenized/data /path/to/output/file_list.txt
```

### 3. Actualización de Configuración
Apunta tu archivo de configuración YAML (en `configs/`) a `file_list.txt`.

---

## 🚀 Cómo Ejecutar

SPES utiliza una arquitectura de **Cliente-Servidor**:
1.  **Servidor de Parámetros:** Gestiona la sincronización de expertos.
2.  **Clientes de Entrenamiento:** Nodos independientes que realizan entrenamiento local.

### ⚙️ Configuración
Parámetros clave de SPES en tu configuración YAML:

```yaml
using_spes: true
spes_config:
  num_peers: 4                  # Total training nodes
  peer_id: 0                    # Current node ID (0-indexed)
  num_train_experts_per_node: 2 # Local experts per node
  sync_steps: 100               # Sync frequency
  server_addr: 127.0.0.1:50051  # Parameter Server Address
```

### Opción A: Lanzamiento Manual (Paso a Paso)

**1. Iniciar Servidor de Parámetros**
```bash
bash run_scripts/run_parameter_server.sh
```

**2. Iniciar Clientes de Entrenamiento (En cada nodo)**
```bash
# Example: Launching on Node 1
bash run_scripts/run_single_node.sh 1

# Optional: Resume from checkpoint
bash run_scripts/run_single_node.sh 0 --resume
```

### Opción B: Lanzamiento en Clúster (Automatizado)

Para SLURM u otros planificadores donde `RANK`, `MASTER_ADDR` y `NPROC_PER_NODE` se configuran automáticamente:

```bash
bash run_scripts/run_cluster.sh
```
*Este script maneja automáticamente el inicio del servidor en el Rank 0 y aísla DDP al nodo local.*

---

## 📊 Evaluación

### 1. Conversión de Puntos de Control
Convierte los puntos de control FSDP fragmentados al formato de HuggingFace:

```bash
# Syntax: <RUN_DIR> <SAVE_STEP> <MODEL_SIZE>
bash eval_scripts/convet_model_to_hf_unshard.sh output/spes_moe_3b_9b/node0 10000 A3B-9B
```

### 2. Ejecución de Benchmarks
Evalúa utilizando `lm-evaluation-harness`:

```bash
bash eval_scripts/eval_full.sh <MODEL_PATH> <MODEL_NAME>
```

---

## 📧 Contacto

¡No dudes en abrir un issue o enviarnos un correo electrónico si tienes alguna pregunta!

**Email:** [jin-rui.zhang@connect.polyu.hk]

---

## 📝 Cita

Si encuentras SPES útil en tu investigación, por favor considera citarlo:

```bibtex
@article{zhang2026pretraining,
  title={Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm},
  author={Zhang, Jinrui and Xiao, Chaodong and Wu, Aoqi and Zhang, Xindong and Zhang, Lei},
  journal={arXiv preprint arXiv:2602.11543},
  year={2026}
}
```

## 🙏 Agradecimientos

Este proyecto se construye sobre los hombros de gigantes. Agradecemos explícitamente los siguientes proyectos y equipos:

*   **[OLMo (Allen Institute for AI)](https://github.com/allenai/OLMo):** Nuestra base de código se construye sobre el excelente código de modelado, entrenamiento e inferencia proporcionado por el equipo [Ai2](https://allenai.org/).
*   **[MegaBlocks (Databricks)](https://github.com/databricks/megablocks):** Utilizamos MegaBlocks para un entrenamiento eficiente de Mezcla de Expertos (MoE) "sin descartes" y operaciones dispersas.
*   **[LM Evaluation Harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness):** Utilizado para nuestro marco de evaluación few-shot y benchmarking.

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia Apache 2.0**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
