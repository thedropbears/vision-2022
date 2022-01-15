#store command info in lists

COMMANDS = ["help", "reload", "local", "var", "debug"]

COMMAND_HELP = [ "help  -  displays this list", "reload <filename> <variable name> <filename to reload>  -  reloads that file for the specified program running on the robot", "local <command>  -  runs any commands intended for the robot on this machine", "var ( list | get | set | reset ) <filename> [variable name]  -  allows the user to access and edit variables on the robot", "debug <filename> [True/False]  -  enables or disables debugging on the specified program if a boolean state is specified, otherwise returns the debug state of that program",]