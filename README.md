# GAN

Repository for GANs (families of generative adversarial network) by Keras (tensorflow)

## requires

- Keras (2.3.1)
- Tensorflow (1.14.0)
- Numpy (1.16.4)
- matplotlib (3.1.0)

All the scripts are refering to https://github.com/eriklindernoren/Keras-GAN . I modified them for explaining the function of the models in detail. Full description is shown in qiita website (but sorry in Japanese).  

## GAN

[今さら聞けないGAN（1）　基本構造の理解](https://qiita.com/triwave33/items/1890ccc71fab6cbca87e) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72490217-f876c700-3859-11ea-9693-7819307773a0.jpeg" width="65%">

<img src="https://user-images.githubusercontent.com/36175603/72490399-85ba1b80-385a-11ea-9970-39730cb1d505.png" width="55%">


----------------------------------

## DCGAN

[今さら聞けないGAN （2）　DCGANによる画像生成](https://qiita.com/triwave33/items/35b4adc9f5b41c5e8141) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72490483-aedaac00-385a-11ea-8911-6e3838e274c5.png" width="55%">


[今さら聞けないGAN（3）　潜在変数と生成画像](https://qiita.com/triwave33/items/a5b3007d31d28bc445c2) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72490592-f6f9ce80-385a-11ea-9e23-8b16dd5d36d8.jpeg" width="55%">

------------------------------


## WGAN-gp

[今さら聞けないGAN（4） WGAN](https://qiita.com/triwave33/items/5c95db572b0e4d0df4f0) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72490757-696aae80-385b-11ea-99e1-c4d45f0711f7.jpeg" width="65%">


[今さら聞けないGAN (5) WGAN-gpの実装](https://qiita.com/triwave33/items/72c7fceea2c6e48c8c07) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72491026-28bf6500-385c-11ea-92be-ce26c4c0befb.gif" width="55%">


--------------------------------


## Conditional GAN


[今さら聞けないGAN（6） Conditional GANの実装](https://qiita.com/triwave33/items/f6352a40bcfbfdea0476) on qiita (in Japanese)

<img src="https://user-images.githubusercontent.com/36175603/72491284-e9dddf00-385c-11ea-883f-a023f863f986.png" width="65%">

<img src="https://user-images.githubusercontent.com/36175603/72491423-5062fd00-385d-11ea-9ff5-12366ac0859c.png" width="50%">


[今さらGAN 聞けないGAN（7）conditional GANの生成画像](https://qiita.com/triwave33/items/d94e5291d45e1d2bdd40) on qiita (in Japanese)

I hacked latent space flag from {0,0,0,0,...,0} to {0,0,0,1,...,0} so that genarated images are changed to "3".

<img src="https://user-images.githubusercontent.com/36175603/72491549-b2bbfd80-385d-11ea-83c9-049339655838.gif" width="50%">

