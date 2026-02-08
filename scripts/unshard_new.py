from olmo_core.distributed.checkpoint import unshard_checkpoint
import os
import shutil


def unshard(input_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path, optim_path = unshard_checkpoint(
            os.path.join(input_dir, 'model_and_optim'),
            output_dir,
            optim=False  # Skip optimizer state
        )
        shutil.copy(os.path.join(input_dir, 'config.yaml'), output_dir)
        print(f'Successfully unsharded checkpoint to: {model_path}')
    except Exception as e:
        print(f'Error unsharding checkpoint: {e}')
        exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="unshard.py", description="Unshard sharded checkpoints on CPU")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    unshard(args.input_dir, args.output_dir)