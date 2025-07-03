## Project Datasets

This project requires two packet data trace files to run:

*   `packet_array_train.pkl`: Used for training the agent.
*   `packet_array_eval.pkl`: A separate file used for evaluating the agent's performance.

Due to their size, these data files are not included in this repository.

### Data Format

The required data should be a NumPy structured array saved as a pickle file. The structure should be:

```python
dtype=[
    ('session_id', 'u4'),   # 4-byte unsigned integer
    ('time', 'f8'),         # 8-byte float (in seconds)
    ('Size_KB', 'f4'),      # 4-byte float
    ('PDUType', 'u1'),      # 1-byte unsigned integer (0=DL, 1=UL)
    ('qfi', 'u1')           # 1-byte unsigned integer
]
The array must be sorted by the 'time' column.
