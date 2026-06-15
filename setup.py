from setuptools import find_namespace_packages, setup


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="callm",
    version="0.1.0",
    description="Continual learning for LLMs with LoRA adapters and prototype routing.",
    author="CaLLM authors",
    url="https://github.com/Bilgecelik/CaLLM-code",
    packages=find_namespace_packages(include=["callm*", "baselines*"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="large-language-models continual-learning automl lora peft",
)
