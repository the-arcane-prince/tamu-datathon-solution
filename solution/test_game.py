import game

async def test_game_1():
    res = await game.model( 
        ["HULU", "NETFLIX", "PEACOCK", "PRIME", "KETCHUP", "MAYO", "RELISH", "TARTAR", "BLUE", "DOWN", "GLUM", "LOW", "GREEN", "MUSTARD", "PLUM", "SCARLET" ], 
        0, 
        False, 
        [  "HULU",
            "NETFLIX",
            "PEACOCK",
            "PRIME"
        ],
        [  
            "HULU",
            "NETFLIX",
            "PEACOCK",
            "PRIME"
        ],None)
    
    print(res)