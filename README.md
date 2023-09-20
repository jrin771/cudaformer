# cudaformer

Implement both the encoder and decoder part of the transformer algorithm. 

The GPT-series only uses the decoder part. 

I'm also interested in implemented flash-attention v1 and v2, but since my AWS AMI uses a Tesla M60 we'll see how far I can get. 

I think even if I can't do the exact low-level hardware optimizations I could still show how to make attention IO-aware, since that is a very important concept for systems optimization. 


