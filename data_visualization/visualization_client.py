# Dummy client that is sending CSI data packet to the server

import os
import socket
import struct
import argparse

def run_client(filemane:str,host:str,port:int):
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    if not os.path.exists(filemane):
        print("File does not exist")
        return
    
    with open(filemane,'rb') as f:
        f.seek(0,os.SEEK_END)
        length=f.tell()
        print(f"file Length:{length}")
        
        f.seek(0,0)
        if struct.unpack("B",f.read(1))[0]==255:
            print("file is Big_Endian format")
            endian='>'
        
        else:
            print("file is Little_Endian format")
            endian='<'
        while f.tell()<length:
            previous=f.tell()
            block_length=struct.unpack(endian+"H",f.read(2))[0]
            block_length+=2
            f.seek(previous,0)
            data_block=f.read(block_length)
            sock.sendto(data_block,(host,port))
            input("Press enter to continue further")
            
            
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dummy client that is sending CSI data packet to the server\n",
        prog="python run_test_client.py"
    )

    parser.add_argument("filename", help="path to the file with a sample data", type=str)
    parser.add_argument("--host", help="host ip to send data", default="127.0.0.1", type=str)
    parser.add_argument("--port", help="host port", default=1234, type=int)

    return parser


if __name__ == "__main__":
    parser = init_argparse()  # Initialize args parser
    args = parser.parse_args()  # Read arguments from command line

    run_client(args.filename, args.host, args.port)

            