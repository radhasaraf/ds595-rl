This folder contains code for implementing the DQN algorithm and its variants
for training the Breakout Atari game

1. Install `requirement.txt`

    ```
    pip install -r requirement.txt
    ```

Carry out either of the steps 2 or 3

2. Head to http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
   and download the .rar file.\
   Extract the .rar file and place the Roms directory required by the Atari
   wrapper in your repo.\
   Run the following:
    ```
    python -m atari_py.import_roms <path to Roms folder>
    ```

3. Run the following commands (Use '!' before each if running in collab)
    
    ```
    pip install ale-py
    pip install automrom
    AutoROM
    ```
