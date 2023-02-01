from paramiko import SSHClient, AutoAddPolicy
from tqdm import tqdm

host = "hpcgpu11"
username = "zuern"
password = "Ahemodad361=?"

client = SSHClient()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(host, username=username, password=password)

with open("tracking_files_hpcgpu11.txt", "r") as f:
    fnames = f.readlines()

ftp_client = client.open_sftp()

for fname in tqdm(fnames):
    fname = fname.strip()

    folder_name = fname.split('/')[-2]
    dst = '/data/argoverse2-full/tracking-results/' + folder_name + '_tracking.pickle'

    ftp_client.get(fname, dst)
ftp_client.close()


















