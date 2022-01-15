import numpy as np
import lt_constants as const

#local variables
do_debug = True

#locate necessary files
magic_num = "magic_numbers.py"
vis = "vision.py"

class cmd_input():
    def __init__(self):
        global magic_num, vis
        #self.magic_defaults = open(magic_num, "r").read()
        #self.vis_defaults = open(vis, "r").read()
        self.testfile = open("lt_constants.py", "r").read()
    def command_interpreter(self,command):
        tokens = command.split(" ")
        if do_debug:
            print(f"[DEBUG] parent command: {command} --> {tokens[0]}")
            for i in range(len(tokens)-1):
                print(f"[DEBUG] arg{i+1}: {command} --> {tokens[i+1]}")
        func_to_call = getattr(self, tokens[0])
        func_to_call(tokens)
    def help(self, args):
        if do_debug:
            print(f"[DEBUG] function help called with tokens of {args}")
        print("-----HELP-----")
        for i in const.COMMAND_HELP:
            print(i)
    def reload(self, args):
        target_file = args[1]
        reload_file = args[2]
        # send command over robot connection to specified file with file to reload
    def local(self,args):
        command = args[1]
        if command == "reload":
            var_name = args[2]
            reload_file = args[3]
            locals()[var_name] = open(reload_file, "r").read()
            if do_debug:
                print(f"[DEBUG] reloaded file {reload_file} in variable {var_name}")
        
cmdtest = cmd_input()

while True:
    cmdtest.command_interpreter(input("> "))