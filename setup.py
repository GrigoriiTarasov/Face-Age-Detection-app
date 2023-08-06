import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

'''with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")'''

setuptools.setup(
    name="face_extrac2",
    version="0.8.",
    author=" ",
    author_email=" ",
    description="API Service to detect faces and esteemate age",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages('face_extrac2'),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.5.5",
    #install_requires=requirements,
)
