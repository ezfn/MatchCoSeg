from pathlib import Path
root_dir = '/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses'
root_path = Path(root_dir)
dirs = root_path.glob('*')
prefixes_to_maintain = ['rectifiedPatchesSIFT_128X128_withField',
                        'frame_']

for currentDir in dirs:
    print('In directory: ' + currentDir.as_posix())
    all_files = currentDir.glob('*')
    for f in all_files:
        do_maintain = False
        for pref in prefixes_to_maintain:
            if pref in f.as_posix():
                do_maintain = True
                break
        if not do_maintain:
            f.unlink()
            pass

