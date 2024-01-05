### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:8501.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

### /!\ Warning for Mac users

When tested on a Mac, the container did not run properly even if the build was completed. This is due to the inability of the docker container to access the camera on MacOS. It gives the following error : 

<img width="866" alt="Capture d’écran 2024-01-05 à 18 15 12" src="https://github.com/Florent-LC/Morphing/assets/96991673/e7b9b295-da6d-404a-9e8c-e705c33885dc">

The following stackoverflow thread gives instructions on how to access the webcam from a Mac : 

https://stackoverflow.com/questions/33985648/access-camera-inside-docker-container/64634921#64634921

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)
