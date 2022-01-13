from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import argparse

files = ["magic_numbers.py", "connection.py", "camera_manager.py"]

parser = argparse.ArgumentParser(
    "Upload code to the Pi. IP defaults to 10.47.74.11"
)
parser.add_argument("-i", "--initial", help="Set pi to use Python", action="store_true")
parser.add_argument("-ip", "--ip", help="Specify a custom ip")
args = parser.parse_args()

main_file = "vision.py"

server_ip = "10.47.74.11" if args.ip is None else args.ip
username = "pi"
password = "raspberry"

ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())
print(f"Connecting to the pi at {server_ip} ... ", end="")
ssh.connect(server_ip, username=username, password=password)
print("Done")

print("Turning off vision ... ", end="")
ssh.exec_command("sudo svc -d /service/camera")
print("Done")

print("Making file system writable ... ", end="")
stdout, stdin, stderr = ssh.exec_command(
    "sudo mount -o remount,rw / ; sudo mount -o remount,rw /boot"
)
for line in stderr:
    print(line)
exit_status = stdout.channel.recv_exit_status()
if exit_status != 0:
    print(f"Something's gone wrong! Error exit status: {exit_status}")
    quit()
else:
    print("Done")

print("Uploading files ... ", end="")
scp = SCPClient(ssh.get_transport())
if args.initial:
    scp.put("runCamera")
    ssh.exec_command("chmod 755 runCamera")
scp.put(files, recursive=True)
scp.put(main_file, remote_path="~/uploaded.py")
print("Done")

print("Making file system read-only ... ", end="")
ssh.exec_command("sudo mount -o remount,ro / ; sudo mount -o remount,ro /boot")
print("Done")

print("Turning on vision ... ", end="")
ssh.exec_command("sudo svc -u /service/camera")
print("Done")

scp.close()