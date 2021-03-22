# Deploy the fibonacci app to manage
You need to deploy an app for the Locust Autoscaler to manage:  
* Build the example app image.  
`docker build -t fibonacci ./app`  
* Deploy the app using a deployment.  
`kubectl apply -f ./app/deployment.yaml`  

