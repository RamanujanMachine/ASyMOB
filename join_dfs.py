import pandas as pd
from pathlib import Path
import sys
import zipfile
import os

DEL_CORRUPTED = True


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    output = Path(sys.argv[2])
    print(f'joining dataframes from {folder} to {output}')

    dfs = []
    for i, fn in enumerate(folder.iterdir()):
        if fn.suffix == '.xlsx':
            try:
                dfs.append(pd.read_excel(fn))
            except zipfile.BadZipFile as e:
                print(f'Found corrupted file! {fn}')
                if DEL_CORRUPTED:
                    print(f'Deleting {fn}')
                    os.remove(fn)
            except Exception as e:
                print('wtf', fn, e)
        if i % 100 == 0:
            print(f'Processed {i} files')   
    result = pd.concat(dfs, ignore_index=True)
    print('Dataframes joined successfully')

    result.to_excel(output, index=False)
    print('Done!')