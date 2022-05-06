# StyleGANDiscriminator
## StyleGANDiscriminator？？☕️
自身の研究で、[genforce/idinvert_pytorch][idinvert_pytorch]を使用させていただいている。しかし、Discriminatorが存在しないため、作成する必要があった。しかし、一からモデルを書くのは大変であるため、[NVlabs/stylegan3][stylegan3]にアップされているpytorch版stylegan2のdiscriminatorを使用させていただいた。ただ、モデルのみであり、重みがない状態であった。そこで、[genforce/idinvert][idinvert]の重みをpytorchモデルに移植すれば良いのではないかという発想に至り、試行錯誤して作成したのが、こちらのStyleGANDiscriminatorである。（都合よくモデルを書き換えただけであるが…）
このリポジトリは、モデルの枠組みしかないため、[watarumacaron/pkl2pt][pkl2pt]を使用して重みを変換した上で使用することをおすすめする。

[idinvert_pytorch]:https://github.com/genforce/idinvert_pytorch
[idinvert]:https://github.com/genforce/idinvert.git
[stylegan3]:https://github.com/NVlabs/stylegan3
[pkl2pt]:https://github.com/watarumacaron/pkl2pt
