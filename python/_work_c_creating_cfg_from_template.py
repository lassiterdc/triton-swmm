#%% import libraries
from string import Template
import pathlib
from __filepaths import *

#%% use template to create new file
with open(f_template_cfg_local, 'r') as T:
    template = Template(T.read())
    # d = {"YEAR":year}
    new_in = template.safe_substitute(d_input)
    # new_file = f_cfg
    new_file_path = pathlib.Path(f_cfg)
    # create _inputs folder if it doesn't already exist
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    # new_file_path.touch()
    with open (f_cfg, "w+") as f1:
        f1.write(new_in)