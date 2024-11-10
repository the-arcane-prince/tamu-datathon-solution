from .trained_llm import TrainedLLM


def test():
    tllm = TrainedLLM()
    print(tllm.run([ "HULU", "NETFLIX", "PEACOCK", "PRIME", "KETCHUP", "MAYO", "RELISH", "TARTAR", "BLUE", "DOWN", "GLUM", "LOW", "GREEN", "MUSTARD", "PLUM", "SCARLET" ], [[  "HULU",
            "NETFLIX",
            "PEACOCK",
            "PRIME"
],[
            "BLUE",
            "DOWN",
            "GLUM",
            "LOW"
        ]
    ], [[  "HULU",
            "NETFLIX",
            "PEACOCK",
            "PRIME"
],[
            "BLUE",
            "DOWN",
            "GLUM",
            "LOW"
        ]
     ]))