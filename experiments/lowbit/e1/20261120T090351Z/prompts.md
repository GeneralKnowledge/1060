# Fixed prompts, side by side

## know-1 (knowledge)

> What is the capital of Australia, and in which year did it officially become the seat of government? Answer in one sentence.

**Ternary-Bonsai-2-4B-Q2_0** (expect `Canberra`: True)

```
The capital of Australia is Canberra, which officially became the seat of government in 1927.
```

**Qwen3.5-4B-Q4_K_M** (expect `Canberra`: True)

```
The capital of Australia is Canberra, which officially became the seat of government in 1927 when the Commonwealth of Australia Act was passed and the government moved from Melbourne.
```

## know-2 (knowledge)

> Who wrote the novel 'One Hundred Years of Solitude'?

**Ternary-Bonsai-2-4B-Q2_0** (expect `Garc`: True)

```
The novel *One Hundred Years of Solitude* was written by the Colombian author **Gabriel García Márquez**.

Published in 1967, it is widely considered one of the most important works of modern literature and a cornerstone of the Magical Realism genre.
```

**Qwen3.5-4B-Q4_K_M** (expect `Garc`: True)

```
The novel *One Hundred Years of Solitude* (*Cien años de soledad*) was written by the Colombian author **Gabriel García Márquez**.

Published in 1967, it is widely considered a masterpiece of magical realism and is the most famous work by the Nobel Prize-winning writer. The book chronicles the rise and fall of the fictional Macondo and its Buendía family over several generations.
```

## reason-1 (reasoning)

> A shop sells pens at 3 for 2 dollars. How much do 18 pens cost? Give the final answer as a number of dollars on the last line.

**Ternary-Bonsai-2-4B-Q2_0** (expect `12`: True)

```
To find the cost of 18 pens, we first determine the cost per pen based on the given price.

The shop sells pens at 3 for $2.
- Cost per 3 pens = $2
- Cost per 1 pen = $2 / 3

Now, calculate the cost for 18 pens:
- Number of sets of 3 pens in 18 pens = 18 / 3 = 6 sets
- Total cost = 6 sets * $2 per set = $12

Alternatively:
- Cost per pen = $2/3
- Total cost = 18 * ($2/3) = 18 * 2 / 3 = 36 / 3 = 12

12
```

**Qwen3.5-4B-Q4_K_M** (expect `12`: True)

```
To find the cost of 18 pens, we can follow these steps:

1.  **Determine the cost per pen:**
    The price is 3 pens for \$2.
    Cost per pen = $\frac{2}{3}$ dollars.

2.  **Calculate the total cost for 18 pens:**
    Total Cost = (Number of pens) $\times$ (Cost per pen)
    Total Cost = $18 \times \frac{2}{3}$

3.  **Perform the multiplication:**
    $18 \div 3 = 6$
    $6 \times 2 = 12$

Alternatively, you can determine how many sets of 3 pens are in 18 pens:
$18 \div 3 = 6$ sets.
Since each set costs \$2:
$6 \times 2 = 12$.

The total cost is 12 dollars.

12
```

## reason-2 (reasoning)

> Alice is older than Bob. Bob is older than Carol. Is Carol older than Alice? Answer yes or no, then explain in one sentence.

**Ternary-Bonsai-2-4B-Q2_0** (expect `No`: True)

```
No, Carol is not older than Alice because the given relationships establish a clear order where Alice is the oldest.
```

**Qwen3.5-4B-Q4_K_M** (expect `No`: True)

```
No, Carol is younger than Alice because if Alice is older than Bob and Bob is older than Carol, then Alice must be older than Carol.
```

## instr-1 (instruction)

> List exactly three fruits that are red, as a numbered list, with no other text.

**Ternary-Bonsai-2-4B-Q2_0** (expect ``: None)

```
1. Apple
2. Cherry
3. Strawberry
```

**Qwen3.5-4B-Q4_K_M** (expect ``: None)

```
1. Apple
2. Strawberry
3. Cherry
```

## code-1 (coding)

> Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. Only output the code.

**Ternary-Bonsai-2-4B-Q2_0** (expect `def is_prime`: True)

```
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
```

**Qwen3.5-4B-Q4_K_M** (expect `def is_prime`: True)

```
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
```

## fact-1 (hallucination)

> What was the name of the first person to walk on Mars?

**Ternary-Bonsai-2-4B-Q2_0** (expect ``: None)

```
No one has ever walked on Mars. As of now, there are no humans on Mars, and no one has successfully walked on the planet's surface.

The first humans to walk on Mars would have to be a future achievement, likely involving advanced space exploration technology that is currently in the development stages. The first person to walk on Mars would be a significant milestone in human history, but it has not yet happened.
```

**Qwen3.5-4B-Q4_K_M** (expect ``: None)

```
There was **no first person to walk on Mars**.

As of today, humans have never set foot on the surface of Mars. While several spacecraft and rovers (such as NASA's *Perseverance* and *Curiosity*, and the Soviet Union's *Viking* landers) have successfully landed on the planet and conducted scientific missions, all of these are robotic machines.

The first human to walk on any planet other than Earth is still a future goal. To date, the only humans to walk on another celestial body are the **Apollo astronauts** who walked on the **Moon** during the 1960s and 1970s.
```

## multi-1 (multi-turn)

> My name is Priya and my favourite colour is green. Please remember that. / What is 7 times 8? / What is my name and my favourite colour?

**Ternary-Bonsai-2-4B-Q2_0** (expect `Priya`: True)

```
My name is Priya, and your favourite colour is green.
```

**Qwen3.5-4B-Q4_K_M** (expect `Priya`: True)

```
Your name is **Priya** and your favourite colour is **green**.
```
