import os
import random
import traceback

from Dataset.CSL import CSL
from Globals import *
from Mesher import *
from Mesher import handle_meshes
from Trainer import Trainer


def main():
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(path.join(output_path, 'models', ''), exist_ok=True)
    os.makedirs(path.join(output_path + 'checkpoints', ''), exist_ok=True)
    os.makedirs(path.join(output_path + 'datasets', ''), exist_ok=True)

    print(f'Loading: input:{args.input} output:{output_path}')

    with StatsMgr.timer('load_data'):
        csl = CSL.from_csl_file(args.input)

    csl.to_ply(path.join(output_path, 'input_csl.ply'))
    csl.to_file(path.join(output_path, 'input.csl'))

    StatsMgr.setitem('n_slices', len([p for p in csl.planes if not p.is_empty]))
    StatsMgr.setitem('n_edges', len(csl))
    print(f'csl={csl.model_name}, slices={StatsMgr["n_slices"]}, edges={StatsMgr["n_edges"]}')

    print(f'Initializing...')

    with Pool(n_process) as pool:
        with StatsMgr.timer('get_datasets'):
            data_sets_promises = csl.get_datasets(pool, len(args.epochs_batches)) if sum(args.epochs_batches) > 0 else None

        trainer = Trainer(csl)
        with open(output_path + 'hyperparams.json', 'w') as f:
            f.write(json.dumps(args, default=lambda o: o.__dict__, indent=4))

        if sum(args.epochs_batches) > 0:
            with StatsMgr.timer('total_train'):
                print(f'Starting train cycle:')
                trainer.train_cycle(data_sets_promises)

            trainer.log_train_losses()

    print(f'Extracting finale mesh...')
    handle_meshes(trainer, 'last')


if __name__ == '__main__':

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    try:
        with StatsMgr.timer('main'):
            main()

    except Exception as e:
        print('X' * 50)
        print(f"an error has occurred: {e}")
        traceback.print_exc()
        print('X' * 50)

    finally:
        stats_str = StatsMgr.get_str()
        with open(output_path + 'stats.json', 'w') as f:
            f.write(stats_str)
        print(f'Done. Artifacts at:{output_path}')
