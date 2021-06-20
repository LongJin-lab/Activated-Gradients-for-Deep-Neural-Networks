# Activated-Gradients-for-Deep-Neural-Networks
This is an offitial implementation of the paper "Activated Gradients for Deep Neural Networks"

## Useage
Simply put "GAF.py" in your main file path, and add this line in the hesd of your training script:

``` from GAF import SGD_atanMom, SGD_atan, Adam_atan, SGD_atanMom_Ada, SGD_tanh_Mom, SGD_log_Mom, SGD_ori```

Change the optimizer as

``` optimizer = SGD_atanMom(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta) ```

or 

``` optimizer = SGD_atan(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta)  ```

Run your code. 
