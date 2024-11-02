## Blunder

Online chess platforms have experienced significant growth, paralleled by an [increase in cheating incidents that undermine the integrity of the game](https://www.chess.com/article/view/online-chess-cheating).

Existing detection systems often struggle to keep pace with the evolving methods of cheating, which range from using chess engines, consulting databases, or receiving external help during play.

There is a surprising lack of research detailing which features and methods are best suitable to detect use of computers in online chess.

Save losses/accs for each model in NPZ and model against Stanford

chess.com monitors if a user is changing tabs and making better moves.

### Marking

The project is assessed on the basis of a written report which should typically contain:

- Title page with abstract (a one or two paragraph summary of the contents).
- Introduction and synopsis, in which the project topic is described and set in the context of published literature, and the main results are briefly summarised.
- Discussion of the work undertaken, in which the various sub-problems, solutions and difficulties are examined.
- If appropriate, a description of experiments undertaken, a presentation of the data gleaned from them, and an interpretation of that data.
- Conclusion, in which the main achievements are reviewed, and unsolved problems and directions for further work are presented.
- Bibliography.

### Objective

The primary objective of the dissertation is to develop a model for detecting and classifying cheating in online chess. This involves:

1. Analysing various cheating methods in online chess.
2. Designing algorithms capable of detecting subtle and complex forms of cheating.
3. Evaluating the effectiveness of these algorithms under different game conditions.

### Research Questions

1. What are the most prevalent methods of cheating in online chess as of the latest data?
2. How can machine learning models be tailored to effectively recognise patterns indicative of cheating?
3. What features or player behaviours are most indicative of cheating and how can they be quantitatively measured?

### Methodology

The project will adopt a mixed-methods approach:

- **Quantitative Analysis:** Using datasets from online chess platforms to analyse patterns and behaviours typical of cheating versus fair play. This will include statistical analysis and machine learning models.
- **Qualitative Analysis:** Interviews and surveys with game developers and players to gain insights into the perceived prevalence and types of cheating, as well as attitudes toward various detection strategies.

### Expected Outcomes

The research is expected to contribute to the development of more sophisticated and fair cheat detection systems for online chess platforms. By improving cheat detection algorithms, we can enhance the credibility and competitive fairness of online chess.

### Key Features

- **Time to Move:** See if all high-quality moves take a consistently similar amount of time.
  - Also check for response time (especially in critical/complex positions) that may suggest time takes to consult engine.
- **Consistency Over Previous Games:** Compare player’s performance in current game to their last 10 to check for sudden improvements.
- **Move Quality Analysis:** Use statistical analysis to evaluate the quality of moves based on chess engine evaluations -> high correlation = likely cheating
- **Player Rating Fluctuations:** Monitor unusual fluctuations in player ratings over a short period of time.
- **Inconsistency in Complexity:** Analyse if a players game shows sporadic spikes in complexity or depth which doesn’t align with previous play.

## Questions

- Model on per-move or per-game basis?
- How to add metadata (time, rating) to convolutions?
- Is it worth including computer vs computer in data?
- Output of LSTM is prediction at next time step IE the outputs of convs for the **next move**, is this useful?
- Are moves independent? Do i need to train on an entire game recursively or just batch individual moves with labels?
- Will cropping games (IE 40 x 12 x 8 x 8) really lead to worse performance?
- 2 dim output with sigmoid or 4 dim (1 for each class) with cross entropy?
- Loss goes to 0, overfitting?
- ConvLSTM loss ends higher than other model - longer training epochs?
-

### Datasets

- [ficsgames](https://www.ficsgames.org)

### Preprocessing

### Board Representation

As per the original paper, the data will be formed into a `nx8x8x6` tensor, where `n` is the number of moves in the game and the 6 channels represent each piece type, with 1 for white and -1 for black.
Can also look into how Alpha-Zero represents the board.

### Model Architecture

Binary cross-entropy loss

Benchmark with linear.

I plan to use convolutional neural networks as the 8x8 chess board lends itself nicely to this architecture.

If a chess board is an image, a sequence of moves is a video.

LSTM over a sequence of moves might be good, but...

> Due to the massive number of possibilities a chess game has, machine learning models cannot use the raw moves of a game as its features. Otherwise, every game is like a new game, and there are no generalisable patterns to learn from.

Conv3d 3x3x3 convs
Transformer
With transformer we can pad and mask the input, so we can utilise all games and the extra empty board states won't affect training as much.

### Chess.com Method

- Engine moves overlap
- Weighting move by importance (opening/endgame moves, fast moves)
- Calculate game accuracy score based on all moves
- Calculate cheat likelihood (based on player elo, elo difference, move times, browser behaviour)

### Test Results

20 epochs at different datasizes

|                   | 1000 | 5000  | 10000 |
| ----------------- | ---- | ----- | ----- |
| **Dense1**        |      | 71.05 | 76.15 |
| **Dense3**        |      | 72.9  | 76.1  |
| **Dense6**        |      | 74.35 | 77.18 |
| **Conv1**         |      | 71.05 | 74.15 |
| **Conv3**         |      | 72.45 | 75.15 |
| **Conv6**         |      | 73.2  | 76.3  |
| **ConvLSTM**      |      | 74.05 | 77.72 |
| **Conv3D**        |      |       | 75.2  |
| **ConvLSTMExtra** |      | 84.9  | 87.98 |

ConvLSTM 30 epochs 78.5%
ConvLSTM 40 epochs 78.67%
ConvLSTMExtra 30 epochs 88.58%

### Stanford Results

|            | 10000 |
| ---------- | ----- |
| **Dense1** | 76.8  |
| **Dense3** | 76.8  |
| **Dense6** | 76.9  |
| **Conv1**  | 77.5  |
| **Conv3**  | 77.8  |
| **Conv6**  | 78.4  |

### Fun Ideas

- Compare heat maps of piece occupation for human & engine

### Stanford Paper (75% Accuracy)

- Only analyses 40 moves (not dynamic)
- Un-balanced dataset

### GitHub Repo (80% Accuracy)

- Only trained on human vs computer (1 side is always cheating)
  - Easier to predict than human vs human with occasional cheater
  - Will always assign 1 to the side cheating more than the other

### Random Quotes

> Cheating is essentially indistinguishable from normal play if it's done cleverly enough at an elite level. This is one of the real challenges, because identifying games with cheating where the cheating consists of only 1 or 2 moves, might be impossible.

> Let's say we are playing a coin flip game. If the coin lands heads, I pay you $1; if tails, you pay me $1. We do this 100 times. If it is a fair coin, this game is statistically a draw. But let's say I am cheating. The coin is fair, except on some of the flips, I use a radio controlled laser thingie to control how the coin will land. As a result, in our 100 flip game, we get 55 tails and I win $5 from you. How would you go about statistically detecting this cheating? It comes down to old fashioned hypothesis testing (H0 = the coin is fair and I'm not influencing it, compute the standard deviation yada yada). Basically you can catch me if and only if I do it so much that my win rate is unlikely to occur naturally. How can you tell which specific flips I'm controlling with my device, even if I do it quite often? You really can't.

### Cheater-Detection Agent

Creating a chess-playing agent (AI) that doesn't just play to win, but also actively tries to identify if the opponent is a human or using computer assistance. The agent would do this by guiding the game into positions that make it easier to tell whether moves are "human-like" or "computer-like."

The chess-playing agent would be designed to steer the game towards board positions that are complex, requiring deeper calculation. In such positions, humans might struggle or make mistakes, while computer-assisted players would more consistently find the optimal moves.

Can use reinforcement learning to maximise the chances of creating positions that reveal cheating behaviour.
**Rewards:** Leading opponent into complex positions, determining how human or computer-like the opponents moves are in these positions.
Can use already trained models for this.

Identify critical positions: analyse past chess games, identifying types of positions where humans tend to make mistakes or play sub-optimally, but computers handle well - can be used as target states.

List((0,List((Drama,8.0), (Action,7.0), (Comedy,7.0))), (1,List((Comedy,9.0), (Drama,9.0), (Action,6.5))), (2,List((Action,8.5), (Drama,8.5), (Comedy,7.5))))

List((0,List((Drama,8.0), (Action,7.0), (Comedy,7.0))), (1,List((Drama,9.0), (Comedy,9.0), (Action,6.5))), (2,List((Drama,8.5), (Action,8.5), (Comedy,7.5))))

### References

- [[Chess Agent Prediction using Neural Networks.pdf]]
- [[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.pdf]]
- [Human and Computer Preferences at Chess](https://cse.buffalo.edu/~regan/papers/pdf/RBZ14aaai.pdf)
- [Performance and Prediction: Bayesian Modelling of Fallible Choice in Chess](https://cse.buffalo.edu/~regan/papers/pdf/HRdF10.pdf)
- [Predictive Modelling of a Chess Player’s Style using Machine Learning](https://nhsjs.com/2024/predictive-modelling-of-a-chess-players-style-using-machine-learning/)
- [Distinguishing between humans and computers in the game of go](https://phys.org/news/2017-11-distinguishing-humans-game.html)
- [On the Limits of Engine Analysis for Cheating Detection in Chess](https://www.researchgate.net/publication/267275282_On_the_Limits_of_Engine_Analysis_for_Cheating_Detection_in_Chess)
- [Similar Project on GitHub](https://github.com/moritzhambach/Detecting-Cheating-in-Chess)
- [Transformers on Images](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1)
- [Transformers on Images Again](https://paperswithcode.com/method/vision-transformer#:~:text=The%20Vision%20Transformer%2C%20or%20ViT,over%20patches%20of%20the%20image.)
- https://paperswithcode.com/task/video-classification
- [The Chess Transformer: Mastering Play using Generative Language Models](https://arxiv.org/pdf/2008.04057)
- [Learning Chess With Language Models and Transformers](https://arxiv.org/abs/2209.11902)
- [Grandmaster-Level Chess Without Search](https://arxiv.org/abs/2402.04494)
- [How the Chess.com Cheat Detection Engine actually works](https://www.youtube.com/watch?v=oTJasAHdu6M)
- [Cheating in Chess wikipedia](https://en.wikipedia.org/wiki/Cheating_in_chess)
  - [Open Spiel](https://github.com/google-deepmind/open_spiel)
- [AlphaZero](https://www.chessprogramming.org/AlphaZero)
