#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/12/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

from utils.argument_parser import get_arguments

def main():
    args = get_arguments()

    print("Task:", args.task)
    print("Model:", args.model)
    print("Data Path:", args.data_path)
    print("Save Path:", args.save_path)
    print("Batch Size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Device:", args.device)

if __name__ == "__main__":
    main()