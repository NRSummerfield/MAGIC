from enum import Enum
from random import choice

BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class color_options(Enum):
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'

    PINK = '\033[95m'
    ORANGE = '\033[93m'
    RED = '\033[91m'

    DEFAULT = '\033[0m'

    @classmethod
    def random(cls):
        return choice(list(cls))

def colorful_print(*inputs, sep=' ', end='\n', color=color_options.DEFAULT, bold: bool=False, underline: bool=False):
    info = sep.join([str(x) for x in inputs])
    line = f'{color.value}{BOLD if bold else ""}{UNDERLINE if underline else ""}{info}{color_options.DEFAULT.value}'
    print(line, end=end)


if __name__ == '__main__':
    colorful_print('This is Black')
    colorful_print('This is Cyan', color=color_options.CYAN)
    colorful_print('This is Green', color=color_options.GREEN)
    colorful_print('This is Blue', color=color_options.BLUE)

    colorful_print('This is Pink', color=color_options.PINK)
    colorful_print('This is Orange', color=color_options.ORANGE)
    colorful_print('This is Red', color=color_options.RED)

    print()
    for i in range(10):
        colorful_print(f'This is a random color! {i=:02d}', color=color_options.random(), underline=True, bold=True)
    


"""
EXAMPLE CODE for a dual print to console (with color!) AND print to a log file

# setting up a logger to track the experiment

base_save_dir = '/path/to/your/experiment'
import logging
logging.basicConfig(filename=os.path.join(base_save_dir, 'Narrative.log'), filemode='a', format='%(asctime)s,%(msecs)d, %(name)s %(levelname)s %(message)s', datefmt="%H:%M:%S", level=logging.INFO)
logging.info("Running training logger")
logger = logging.getLogger('TrainingLog')

def out(*inputs, sep=' ', end='\n', color: color_options = color_options.DEFAULT, bold: bool=False, underline: bool=False):
    logger.info(sep.join([str(x) for x in inputs]) + '')
    colorful_print(*inputs, sep=sep, end=end, color=color, bold=bold, underline=underline)
cout = out

# then just mass replace `print` with `cout`
# Note: You can randomly select a color with `color_options.random()`
"""