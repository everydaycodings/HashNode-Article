---
title: "Understanding Gradient Descent | Episode 6"
seoTitle: "Mastering Gradient Descent: Optimize Machine Learning Models"
seoDescription: "Dive into gradient descent in machine learning. Learn applications, variants, and mathematical foundations. Master optimization for model refinement."
datePublished: Sat Dec 02 2023 15:11:08 GMT+0000 (Coordinated Universal Time)
cuid: clpo6y9ha000908l51utm95pj
slug: understanding-gradient-descent-episode-6
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701247154058/7f8baa11-8211-4879-8160-b5515ec4c9aa.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1701247133366/b2291f2b-acd5-4968-87f2-ba685b5dd347.png
tags: algorithms, data-science, machine-learning, deep-learning, gradient-descent

---

## Introduction

Welcome back! In the [previous episode](https://neuralrealm.hashnode.dev/understanding-linear-regression-episode-5), we were introduced to the cost function j and how different values of the parameters w and b affected the cost. However, it would be helpful to have a more organized approach to determine the values of w and b that result in the smallest cost, j of w, b. Thankfully, we can utilize an algorithm called `gradient descent` to achieve this goal.

Gradient descent is a widely used technique in machine learning, which is not limited to linear regression. It is used to train some of the most advanced neural network models, also known as deep learning models. You learned about deep learning models in the future. Understanding gradient descent will provide you with one of the most crucial building blocks in machine learning.

### Breaking Down the Basics

#### The Journey with Gradient Descent

So, what does gradient descent entail? At its core, you aim to minimize the cost function *J(w,b)*. While we've primarily explored linear regression's cost function thus far, it's essential to note that gradient descent isn't exclusive to specific cost functions. It's a universal algorithm applicable to a broader array of functions, even those involving more than two parameters.

Consider a generalized cost function *J(w1,w2,…,wn,b)*. The objective remains consistent — find values for w1​ through *wn* and *b* that yield the smallest possible *J*. Gradient descent emerges as the algorithm of choice for this task.

#### Initiating the Descent

The journey begins with initial guesses for *w* and *b*. While the choice of initial values can influence the process, for linear regression, a common practice is to set both to 0. With this starting point, the gradient descent algorithm iteratively adjusts *w* and *b* to minimize *J(w,b)*.

#### Unraveling the Descent Process

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701248834779/49b7a6d1-4527-47fe-8b76-7bf2dafaab4b.webp align="center")

Imagine standing atop a hill represented by the cost function surface *J(w,b)*. The goal is to descend efficiently to the valleys, which signify the minima of the cost function. This descent is guided by the direction of the steepest descent, akin to taking tiny steps downhill.

Picture the surface as a hilly landscape where high points are hills and low points are valleys. With each step, you evaluate the terrain, choosing the direction that leads downhill most swiftly. Iteratively, these steps bring you closer to a valley, eventually settling at or near a minimum.

#### The Fascinating Property of Multiple Minima

An intriguing facet of gradient descent is its sensitivity to the starting point. Choosing different initial values for *w* and *b* might lead you to distinct local minima. These local minima are essentially the bottoms of the valleys you reach through the descent process. The algorithm tends to converge to the local minimum dictated by the initial conditions.

## Mathematical Foundation

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701251041007/2d5c293a-adcf-47fc-a157-4f4d63a6b829.png align="center")

If you feel like there's a lot going on in this equation, it's okay, don't worry about it. We'll unpack it together.

Now, this dives more deeply into what the symbols in this equation mean. The symbol `α` is the Greek alphabet Alpha. In this equation, Alpha is also called the learning rate. The `learning rate` is usually a small positive number between `0 and 1` and it might be say, 0.01. What Alpha does is, it basically controls how big of a step you take downhill. If Alpha is very large, then that corresponds to a very aggressive gradient descent procedure where you're trying to take huge steps downhill. If Alpha is very small, then you'd be taking small baby steps downhill. We'll come back later to dive more deeply into how to choose a good learning rate Alpha. Finally, this term `∂J/∂b J(w,b)`, is the derivative term of the cost function `J`. Let's not worry about the details of this derivative right now. But later on, you'll get to see more about the derivative term. But for now, you can think of this derivative term that I drew a magenta box around as telling you in which direction you want to take your baby step. In combination with the learning rate *Alpha, it also determines the size of the steps you want to take downhill.* Now, I do want to mention that derivatives come from calculus. Even if you aren't familiar with calculus, don't worry about it. Even without knowing any calculus, you'd be able to figure out all you need to know about this derivative term in this episode.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701251823176/31fdff13-98b9-4085-a61a-9046e087328c.png align="center")

It's important to remember that your model has two parameters, not just `w`, but also `b`. There's an assignment operation that updates the parameter b, which looks very similar to the one for `w`. In this operation, `b` is assigned the old value of `b minus` the learning rate Alpha times the derivative term d/db of J of wb.

In the surface plot graph, you take baby steps until you reach the bottom value. For the gradient descent algorithm, you repeat these two update steps until the algorithm converges. By `converges`, it means that you've reached the point of a local minimum where the parameters w and b no longer change much with each additional step taken. In semantic gradient descent, you're updating two parameters: w and b. This update takes place for both parameters. It's important to simultaneously update w and b, meaning you want to update both parameters at the same time for gradient descent to work correctly.

### Gradient Descent Intution

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701429564826/b086c21f-52b4-4bef-8dd8-ad5369bdfe92.png align="center")

To better understand the process, let's use a simpler example where we focus on minimizing just one parameter. Imagine you have a cost function J that depends on only one parameter w, which is a number. In this case, gradient descent can be expressed as W = w - (Alpha \* dJ/dw), where Alpha is the learning rate. Essentially, you are trying to minimize the cost by adjusting the parameter w.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701429527907/11e5076d-2e22-4dcb-8952-5c39ed0894cd.png align="center")

The horizontal axis represents parameter w, while the vertical axis represents the cost J of w. We begin by initializing gradient descent with a starting value for w at a particular location on the function J. If we start at a specific point on the function J, the gradient descent will update w to be w minus the learning rate Alpha times d over dw of J of w.

#### Left Diagram

To understand what this derivative term means, we can draw a tangent line at that point. This line touches the curve at that point, and its slope represents the derivative of the function j at that point. The slope of the tangent line can be determined by drawing a small triangle, where the height divided by the width of the triangle gives us the slope. When the tangent line is pointing up and to the right, the slope is positive, meaning that this derivative is a positive number and greater than 0. The updated w is w minus the learning rate times some positive number.

If we take w minus a positive number, we end up with a new value for w that is smaller. This means that we are moving to the left on the graph and decreasing the value of w. This is the right thing to do if our goal is to decrease the cost J because when we move towards the left, the cost J decreases, and we are getting closer to the minimum for J.

#### Right Diagram

Now, let's consider another example. Suppose we take the same function J of w as above, but we initialize gradient descent at a different location. Say by choosing a starting value for w that's over here on the left. That's this point of the function J. The derivative term, remember, is d over dw of J of w, and when we look at the tangent line at this point over here, the slope of this line is the derivative of J at this point. But this tangent line is sloping down into the right. This means that the derivative of J at this point is a negative number.

When we update w, we get w minus the learning rate times a negative number. This means that we subtract from w a negative number. But subtracting a negative number means adding a positive number, and so we end up increasing w. This step of gradient descent causes w to increase, which means we are moving to the right of the graph, and our cost J has decreased down to here.

In summary, gradient descent changes w to get us closer to the minimum. The derivative term in gradient descent makes sense because it helps us to find the direction of the steepest descent, which is the direction that will bring us closer to the minimum. In addition to the derivative term, another key quantity in the gradient descent algorithm is the learning rate alpha. In the next video, we will explore how to choose the value of Alpha, its impact on the performance of the gradient descent algorithm, and how to select a good value for Alpha in your implementation of gradient descent.

## Learning Rate

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701487108872/8af35383-86e9-4487-9724-e886fb106ed3.png align="center")

Let's consider the impact of the learning rate alpha being too small or too large. If the learning rate is too small, gradient descent will still work, but it will be incredibly slow. It will take a lot of tiny baby steps to reach the minimum, which will be a very time-consuming process. This happens because when the learning rate is too small, the derivative term is multiplied by a very small number, i.e., alpha, and as a result, gradient descent takes very small steps towards the minimum, which prolongs the process.

On the other hand, if the learning rate is too large, gradient descent may overshoot and never converge to the minimum. This happens because, in this case, the step taken by gradient descent is huge, which may lead to an acceleration that overshoots the minimum. As a result, gradient descent may end up getting further away from the minimum after each update, making it diverge.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701523953209/211aa03f-6a54-4a11-b2db-e73c8e4c275d.png align="center")

Let me answer another question you may have. You might be wondering if one of your parameters, let's call it W, has already reached a local minimum, is it necessary to take another step of gradient descent?

The answer is no. If you are already at a local minimum, performing further steps of gradient descent will not change the value of W.

To explain why, let's suppose you have a cost function J, which has two local minima, and your current value of W is equal to 5 after some number of steps of gradient descent. This means you are at a point on the cost function J, which happens to be a local minimum. At this point, the slope of the line is zero, and the derivative term is equal to zero for the current value of W. Therefore, the update equation becomes W = W - (learning rate \* 0), which is the same as saying W remains unchanged.

For example, let's say that the current value of W is 5, and the learning rate alpha is 0.1. After one iteration, you update W as W - alpha \* 0, which is still equal to 5. So, if your parameters have already brought you to a local minimum, then further gradient descent steps will do nothing. This is what you want because it keeps the solution at that local minimum.

**This also explains why gradient descent can reach a local minimum even with a fixed learning rate alpha.**

## Gradient Descent For Linear Regression

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527677367/6666789f-4aff-401d-8b13-ee629e77803d.png align="center")

In the previous section, we looked at the linear regression model and cost function, followed by the gradient descent algorithm. In this section, we will use the squared error cost function for the linear regression model with gradient descent to train the model to fit a straight line and achieve the training data.

When we calculate the derivatives, we get the following terms(see the image above). The derivative with respect to W is 1 over m, the sum of i equals 1 through m, multiplied by the error term, which is the difference between the predicted and the actual values, times the input feature xi. The derivative with respect to b is the same as the equation above, except that it doesn't have the xi term at the end. If we use these formulas to calculate these two derivatives and implement gradient descent, it will work.

You might be curious about how we derived these formulas. We derived them using calculus.

#### <mark>T</mark>**<mark>o gain a better understanding of how this process works, I recommend referring to this Notebook</mark>:** [**Gradient Descent Notebook**](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Linear%20Regression/Gradient_Descent.ipynb)

---

### **🎙️ Message for Next Episode:**

In the next episode, we will delve into making linear regression much more powerful than just considering one feature like the size of a house. You will learn how to make it work with multiple features, as well as how to fit nonlinear curves. These improvements will make the algorithm much more valuable and useful. Additionally, we will cover some practical tips that will aid in getting linear regression to work on practical applications. I'm thrilled to have you here in this class with me, and I'm looking forward to seeing you next episode.

---

### **Resources For Further Research**

1. **Coursera** - [**Machine Learning Specialization**](https://www.coursera.org/learn/machine-learning/home/welcome)
    
    This Article is heavily inspired by this Course so I will also recommend you to check this course out, there is an option to watch the course for free.
    
2. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron:**
    
    * A practical guide covering various machine learning concepts and implementations.
        

---

## **By the way…**

#### Call to action

*Hi, Everydaycodings— I’m building a newsletter that covers deep topics in the space of engineering. If that sounds interesting,* [***subscribe***](https://neuralrealm.hashnode.dev/newsletter) *and don’t miss anything. If you have some thoughts you’d like to share or a topic suggestion, reach out to me via* [***LinkedIn***](https://www.linkedin.com/in/kumar-saksham1891/) *or* [***X***](https://twitter.com/everydaycodings).

#### References

*And if you’re interested in diving deeper into these concepts, here are some great starting points:*

* [**Kaggle Stories**](https://neuralrealm.hashnode.dev/series/kaggle-stories) *\-* Each episode of Kaggle Stories takes you on a journey behind the scenes of a Kaggle notebook project, breaking down tech stuff into simple stories.
    
* [**Machine Learning**](https://neuralrealm.hashnode.dev/series/machine-learning) *\-* This series covers ML fundamentals & techniques to apply ML to solve real-world problems using Python & real datasets while highlighting best practices & limits.