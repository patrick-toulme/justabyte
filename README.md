# Just a Byte

Hi! I'm Patrick Toulme, a compiler and performance engineer. I write [Just a Byte](https://patricktoulme.substack.com/) — a blog about AI compilers, silicon, and systems.

This repo contains companion code, IR dumps, and reproduction scripts for the blog posts.

[![Website](https://img.shields.io/badge/Website-patricktoulme.com-blue?style=flat&logo=google-chrome&logoColor=white)](https://patricktoulme.com/) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Patrick%20Toulme-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/patrick-toulme-150b041a5/) [![X](https://img.shields.io/badge/X-@PatrickToulme-black?style=flat&logo=x&logoColor=white)](https://x.com/PatrickToulme)

## Posts

**Post 1:** [From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
Traces 8 lines of JAX through the full TPU compiler pipeline — from HLO through optimization passes to 250 VLIW bundles across five fused kernels.

**Post 2:** [When XLA Isn't Enough: From Pallas to VLIW with Splash Attention on TPU](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
Explores the limits of XLA's automatic optimization for attention and how Pallas custom kernels achieve 6x fewer VLIW bundles and 37x less HBM traffic.

**Post 3:** [CuTile on Blackwell: NVIDIA's Compiler Moat Is Already Built](https://patricktoulme.substack.com/p/cutile-on-blackwell-nvidias-compiler)
Traces a Mixture of Experts kernel through NVIDIA's CuTile stack — 86 lines of Python compiled into 1,900 lines of optimized PTX with tcgen05 instructions.

**Post 4:** [Frontier Pretraining Infrastructure Is Already Open Source: GPT-OSS on TPU with MaxText](https://patricktoulme.substack.com/p/frontier-pretraining-infrastructure)
Shows how MaxText and XLA compress 11,207 HLO instructions into 887 fused kernels, arguing frontier training infra is already available in open source.
