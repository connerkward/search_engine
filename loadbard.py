import sys
import time

LOADBAR_CHAR = "█"

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor
progress = spinning_cursor()

def wheel(message):
    sys.stdout.write(f'\b')
    sys.stdout.write(next(progress))
    sys.stdout.flush()

def bar(message):
    sys.stdout.write(LOADBAR_CHAR)
    sys.stdout.flush()

def percentage(message, replace=False):
    if replace:
        sys.stdout.write(f'\b'*len(message))
        sys.stdout.write(message)
        sys.stdout.flush()
    else:
        print(message)

class LoadBard:
    start_message = ""
    mod_message = ''
    time_start = float()
    timer = False
    log = False
    detailed = False
    TYPES = {"wheel": wheel, "bar": bar, "percentage": percentage}
    update_func = None
    def start(self,message="", timer=False, type="percentage"):
        self.update_func = self.TYPES[type]
        if timer:
            self.timer = True
            self.time_start = time.perf_counter()
        self.start_message = message
        self.mod_message = f'{message}:\t'
        sys.stdout.write(self.mod_message)

    def update(self, message="", replace=False):
        self.update_func(self.mod_message+message, replace=replace)


    def end(self):
        s = "\b"*len(self.mod_message)
        if self.timer:
            sys.stdout.write(f'{s}{self.start_message} completed. '
                             f'Completed in {time.perf_counter()-self.time_start} seconds.')
        else:
            sys.stdout.write(f'{s}{self.start_message} completed.')
        sys.stdout.flush()
        sys.stdout.write("\n") # this ends the progress bar

def log(line):
    with open("log.txt", "a") as f:
        f.write(line + "\n")