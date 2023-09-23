# CUDAFORMER 


What is it? Implementing the transformer architecture on an NVIDIA Tesla M60 GPU. 

Why? James Lin (Transformer-CUDA on github) started doing it a few days ago and I thought it was a great idea to learn AI systems, parallel programming, AWS, CUDA, and to do a cool project.  

What do you have so far? I've implemented multi-head attention (the main building block of the transformer) in CUDA and plan to implement the rest shortly. 









## TO-DO 
I plan to clean this up a lot more in the future. 
## OVERVIEW   
Inspired by James Lin's attempt on this, you should check that out.  


## INSTALLATION  
I'll add steps for AWS since it was my first time using EC2 and that might be helpful for replication purposes.   


AWS:  
Chmod 400 ______.pem  
ssh -i "_____.pem" ubuntu@<IP-address> (or <DNS-address>)


## TROUBLE-SHOOTING  
AWS has good guides, 


## HOW TO RUN PROGRAMS 


nvcc <program-name>.cu -o <program-name> 

./<program-name> 


## LEARNING RESOURCES I USED 

Deeper Understanding of Positional Encoding: https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3 
