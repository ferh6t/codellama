FROM nvidia/cuda:12.2.0-base-ubuntu20.04


ENV FTP_PROXY="http://192.168.0.1:8119"
ENV  HTTPS_PROXY="http://192.168.0.1:8119"
ENV  HTTP_PROXY="http://192.168.0.1:8119"
ENV  NO_PROXY="localhost,127.0.0.1,.test"
ENV  ftp_proxy="http://192.168.0.1:8119"
ENV  http_proxy="http://192.168.0.1:8119"
ENV  https_proxy="http://192.168.0.1:8119"
ENV  no_proxy="localhost,127.0.0.1,.test"

WORKDIR /app
RUN apt-get update && \
    apt-get install -y python3 python3-pip
COPY . /app/

VOLUME ./$MODEL:/app/$MODEL

RUN pip install -e .

EXPOSE 3000

RUN echo '#!/bin/sh' > /app/entrypoint.sh && \
    echo 'torchrun --nproc_per_node 1 example_completion.py --ckpt_dir /app/$MODEL/ --tokenizer_path /app/$MODEL/tokenizer.model --max_seq_len 128 --max_batch_size 4' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set the entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
