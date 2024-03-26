<div align="center">
<h1>LexiCrowd: A Learning Paradigm towards Text to Behaviour Parameters for Crowds</h1>
<strong><a href="https://marilenalemonari.github.io/" target="_blank">Marilena Lemonari</a>, <a href="https://nefeliandreou.github.io/" target="_blank">Nefeli Andreou</a>, <a href="https://www.cs.upc.edu/~npelechano/" target="_blank">Nuria Pelechano </a>, <a href="https://totis77.github.io/" target="_blank">Panayiotis Charalambous</a>, <a href="http://www.cs.ucy.ac.cy/~yiorgos/" target="_blank">Yiorgos Chrysanthou</a>

CLIPE Workshop Eurographics 2024</br>
April 2024</strong>
</div>

![Demo Image](https://github.com/MarilenaLemonari/LexiCrowd/blob/main/Misc/teaser.png)

<p align="justify">
Creating believable virtual crowds, controllable by high-level prompts, is essential to creators for trading-off authoring freedom and simulation quality. The flexibility and familiarity of natural language in particular, motivates the use of text to guide the generation process. Capturing the essence of textually described crowd movements in the form of meaningful and usable parameters, is challenging due to the lack of paired ground truth data, and inherent ambiguity between the two modalities. In this work, we leverage a pre-trained Large Language Model (LLM) to create pseudo-pairs of text and behaviour labels. We train a variational auto-encoder (VAE) on the synthetic dataset, constraining the latent space into interpretable behaviour parameters by incorporating a latent label loss. To showcase our model’s capabilities, we deploy a survey where humans provide textual descriptions of real crowd datasets. We demonstrate that our model is able to parameterise unseen sentences
and produce novel behaviours, capturing the essence of the given sentence; our behaviour space is compatible with simulator parameters, enabling the generation of plausible crowds (text-to-crowds). Also, we conduct feasibility experiments exhibiting the potential of the output text embeddings in the premise of full sentence generation from a behaviour profile.
</p>

<br>

<p align="center"><strong>
	- <a href="https://github.com/MarilenaLemonari/LexiCrowd/blob/main/Misc/CLIPWEG24_paper_4_final.pdf" target="_blank">PDF Paper</a> | <a href="https://youtu.be/xSbMqwq0Mp8" target="_blank">Video</a> -
</strong>
</p>

<br>

<p align="center" dir="auto">
	<a href="https://youtu.be/xSbMqwq0Mp8" rel="nofollow">
		<img align="center" width="400px" src="https://github.com/MarilenaLemonari/LexiCrowd/blob/main/Misc/thumbnail.png"/>
	</a>
</p>
