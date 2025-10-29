import numpy as np

def encodeMessage(message, maskingSignal):
    return (message+maskingSignal)

def decodeMessage(recievedSignal, estimatedSignal):
    return (recievedSignal-estimatedSignal)