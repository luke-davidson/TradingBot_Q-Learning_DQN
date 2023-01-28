$$
\begin{align}
\text{State}~:~S&\in\begin{bmatrix}
[\text{Close prices}]\\
[\text{Open price}]\\
[\text{Day high prices}]\\
[\text{Day low prices}]\\
[\text{Adjusted close prices}]\\
[\text{Volume}]\\
[\text{Position history}]\\
\end{bmatrix}^T\\
\text{Actions}~:~A&\in\{
\text{BUY}, 
\text{HOLD},
\text{SELL}\}\\
\text{Rewards}~:~R(s,a,s')&=\frac{\text{current price}}{\text{last transaction price}} - \text{transaction fee}\\
\text{Transitions}~:~T(s,a)&=\begin{bmatrix}
[\text{Close prices[1:], next close price}]\\
[\text{Open prices[1:], next open price}]\\
[\text{Day high prices[1:], next day high price}]\\
[\text{Day low prices[1:], next day low price}]\\
[\text{Adjusted close prices[1:], next day adj. close price}]\\
[\text{Volume[1:], next day volume}]\\
[\text{Position history[1:], next position}]\\
\end{bmatrix}^T\\
\text{Position}&:\begin{cases}
-1&\text{if SHORT}\\
0&\text{if FLAT}\\
1&\text{if LONG}\\
\end{cases}\\
\text{Next Position}&\gets\begin{cases}
\text{SHORT} &\text{if SHORT and HOLD}\\
\text{SHORT} &\text{if SHORT and SELL}\\
\text{SHORT} &\text{if FLAT and SELL}\\
\\
\text{FLAT} &\text{if FLAT and HOLD}\\
\text{FLAT} &\text{if SHORT and BUY}\\
\text{FLAT} &\text{if LONG and SELL}\\
\\
\text{LONG} &\text{if LONG and HOLD}\\
\text{LONG} &\text{if LONG and BUY}\\
\text{LONG} &\text{if FLAT and BUY}\\
\end{cases}\\
\text{Transaction fee}(a)&\gets\begin{cases}
\text{Trade fee bid percentage}~\times~\text{price}&\text{if}~a=~\text{SELL}\\
\text{Trade fee ask percentage}~\times~\text{price}&\text{if}~a=~\text{BUY}\\
0&\text{if}~a=~\text{HOLD}\\
\end{cases}
\end{align}
$$







DQN

```
Initialize behavior and target neural networks Q and T
for each episode:
    state <- first state
    for each timestep:
        action <- argmax over all action values for Q(s)
        next_state, reward, done <- step(state, action)
        
        if done:
            td_target = reward
        else:
            td_target = reward + (gamma * max(T(next_state)))

        perform gradient descent step to move Q(s)[action] towards td_target
    
```









k
