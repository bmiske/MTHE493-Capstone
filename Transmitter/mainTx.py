from TransmitterBase import TransmitterBase
from transmitter import transmitter
from TransmitterParameters import TransmitterParameters
import numpy as np

params = TransmitterParameters()
top_block = transmitter()

txBase = TransmitterBase(top_block, params)
message = np.random.randint(0, 2, 32)

top_block.start()
txBase.transmitMessage(message)