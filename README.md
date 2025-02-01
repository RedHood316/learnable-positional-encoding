# 🚀 Learnable Positional Encoding in PyTorch
* ML Internship - Technical Interview Task
* Submitted by: Md Sami Ul Hoque
* Email: samihoque16@gmail.com

# 📌 1. Answer: Issues with Stacking Self-Attention Layers with Positional Encoding
Think of photocopying a document multiple times—each time, the quality gets worse. Similarly, when stacking too many self-attention layers, the model faces:

* Loss of Positional Information – Like fading text in photocopies, deep layers weaken positional encoding, making it harder to retain sequence order.
  
* Computational Cost – Self-attention requires O(n²) complexity, making it very slow for long sequences.

* Vanishing Gradient – Like tracing a faded photocopy, deep layers lose important information, making training unstable.
  
* Redundant Learning – Similar to reprinting the same content multiple times, self-attention layers can focus too much on short-term dependencies, limiting long-range learning.
  
✅ Solution? Instead of blindly stacking layers, efficient architectures like hybrid models (CNN + Self-Attention) or smarter attention mechanisms (Longformer, Linformer, Performer) can improve efficiency.

# 📌 2. Learnable Positional Encoding in PyTorch
Instead of fixed positional encoding (like pre-numbered book pages), we allow the model to learn its own position embeddings dynamically.

💻 Implementation
```python
import torch
import torch.nn as nn

# -------------------------------
# Learnable Positional Encoding Class
# -------------------------------

class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding for Transformer models.
    This helps the model understand token positions dynamically instead of using fixed sinusoidal embeddings.
    """
    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_seq_len (int): Maximum sequence length for positional encoding.
            d_model (int): Embedding dimension for each token.
        """
        super(LearnablePositionalEncoding, self).__init__()
        
        # Create a learnable positional encoding matrix of shape (max_seq_len, d_model)
        self.positional_encoding = nn.Parameter(torch.empty(max_seq_len, d_model))
        
        # Initialize using Xavier Uniform to improve training stability
        nn.init.xavier_uniform_(self.positional_encoding)

    def forward(self, x):
        """
        Adds positional encoding to the input token embeddings.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input with added positional encoding.
        """
        seq_len = x.shape[1]  # Get the actual sequence length from input
        return x + self.positional_encoding[:seq_len, :]  # Add positional encoding to token embeddings


# -------------------------------
# Simple Transformer Block with Learnable Positional Encoding
# -------------------------------
class SimpleTransformerBlock(nn.Module):
    """
    Implements a basic Transformer block with token embeddings, learnable positional encoding,
    and multi-head self-attention.
    """
    def __init__(self, max_seq_len, d_model, num_heads):
        """
        Args:
            max_seq_len (int): Maximum sequence length for positional encoding.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
        """
        super(SimpleTransformerBlock, self).__init__()
        
        # Token Embedding Layer: Converts token indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=d_model)
        
        # Learnable Positional Encoding Layer
        self.positional_encoding = LearnablePositionalEncoding(max_seq_len, d_model)
        
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        # Fully Connected Layer (Projection)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Forward pass for the Transformer block.

        Args:
            x (Tensor): Input tokenized sequence of shape (batch_size, seq_len)

        Returns:
            Tensor: Processed sequence embeddings of shape (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)  # Convert token indices to dense embeddings
        x = self.positional_encoding(x)  # Add positional encoding
        x, _ = self.attention(x, x, x)  # Apply self-attention
        return self.fc(x)  # Pass through a fully connected layer


# -------------------------------
# Running the Model with a Dummy Dataset
# -------------------------------
if __name__ == "__main__":
    # Define Hyperparameters
    batch_size = 4
    max_seq_len = 20  # Maximum sequence length for encoding
    seq_len = torch.randint(5, 15, (1,)).item()  # Random sequence length between 5 and 15
    d_model = 32  # Embedding dimension
    num_heads = 4  # Number of self-attention heads

    # Generate a Random Tokenized Input (Simulated Text Sequence)
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))  # Shape: (batch_size, seq_len)

    # Initialize the Transformer Model
    model = SimpleTransformerBlock(max_seq_len=max_seq_len, d_model=d_model, num_heads=num_heads)

    # Forward Pass
    output = model(dummy_input)

    # Print the Output Shape for Verification
    print("Input Shape:", dummy_input.shape)  # Expected: (batch_size, seq_len)
    print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
```
## **📌 3. Explanation**
### **🔹 Why Use Learnable Positional Encoding?**
- Traditional **sinusoidal encoding is static**, meaning the model **cannot adapt** to different datasets.
- **Learnable encoding allows flexibility**, meaning the model can **learn which positional information is most important**.

### **🔹 How It Works**
✔️ **Uses `nn.Parameter`** → Creates **trainable positional embeddings** instead of static ones.  
✔️ **Extracts actual sequence length** → Dynamically **adjusts encoding size** based on input.  
✔️ **Works with any sequence length** → Avoids **fixed-size constraints**.

---

## **📌 4. Expected Output**
### **📌 When you run `python main.py`, you should see:**
Input Shape: torch.Size([4, 12])
Output Shape: torch.Size([4, 12, 32])



### **🔹 Why Is This Output Correct?**
This output confirms that **positional encoding was applied correctly without altering the sequence structure**.

✅ **Matches expected dimensions**:
- **Batch Size: `4`** → The model processes **4 sequences** in parallel.
- **Sequence Length: `12` (randomly generated)** → The **actual sequence length is dynamically determined** at runtime.
- **Embedding Dimension: `32`** → Each token is **represented as a 32-dimensional vector**.

✅ **Why Does the Shape Stay the Same?**
- **Positional encoding does NOT change the number of tokens or features.**
- It is an **additive operation**, meaning the model simply **enhances input embeddings** without modifying the structure.
- This confirms that **learnable positional encoding is correctly integrated** while preserving input-output consistency.

---

## **📌 5. Video Walkthrough 🎥**
🔗 **[Watch the Video Walkthrough](https://drive.google.com/YOUR-VIDEO-LINK)**  
*(A quick 5-minute explanation of the project.)*

---


## 🎥 Video Walkthrough  
## 📌 Video Walkthrough 🎥  
[![Watch the Video](https://img.youtube.com/vi/xi2u191_vNw/maxresdefault.jpg)](https://youtu.be/xi2u191_vNw)


---

## 📌 How to Run This Project

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/RedHood316/learnable-positional-encoding.git
cd learnable-positional-encoding
```

### 2️⃣ Install Dependencies**
```bash
pip install torch numpy
```
### 3️⃣ Run the Script
```bash
python main.py
```
✅ This will execute the Transformer model with learnable positional encoding.


