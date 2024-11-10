import google.generativeai as gemini
import os

class TrainedLLM:
    
    prompt = r"""Connections is a clever, engaging puzzle game that challenges players to group words based on hidden relationships. In each round, players are presented with a grid of 16 words and must identify four unique categories, with each category grouping four words that share a common theme or concept. The challenge lies in spotting subtle associations, distinguishing close but incorrect connections, and using deductive reasoning to avoid errors.

The game tests and sharpens vocabulary, lateral thinking, and pattern recognition, offering a refreshing blend of strategy and wordplay. Each level becomes progressively more complex, making Connections perfect for players who enjoy thought-provoking puzzles and honing their logical skills.

The categories in Connections are often cleverly crafted to challenge players' linguistic intuition and reasoning skills. Here’s how they use subtlety, misdirection, and semantic complexity to elevate the challenge:

Homophones and Similar Sounding Words: Some categories include words that sound alike but have different meanings or spellings (e.g., pair vs. pear vs. pare). This forces players to think beyond the spelling and consider phonetic relationships, often leading to unexpected connections.

Multi-Layered Meanings: Certain words can belong to more than one category based on their different meanings or uses. For example, apple might fit into a category about fruit but could also relate to tech companies. This creates an ambiguity that requires players to think critically about context.

Thematic Misleading Associations: Some categories are intentionally deceptive by including words that seem related but don't quite fit the actual group. For example, lemon, lime, blue, and berry could trick players into thinking it’s about fruits when it’s actually colors.

Hidden Relationships and Conceptual Links: Categories sometimes rely on less obvious, abstract connections, like grouping words that represent emotions, philosophical concepts, or actions (e.g., hope, faith, love, charity). These categories require players to recognize more nuanced themes rather than concrete groups.

Cultural or Idiomatic Associations: Categories may include words that only connect in certain cultural or idiomatic contexts, such as phrases or common expressions (e.g., cold, turkey, wild, goose for "wild goose chase" and "cold turkey"). Recognizing these requires a certain familiarity with idioms or pop culture references.

Linguistic Tricks with Parts of Speech: Words may be grouped based on their grammatical function (like verbs, nouns, or adjectives), even if they have no thematic connection otherwise. For instance, words like run, jump, fly, and dive could lead players to a "sports" misinterpretation when it’s actually about actions or verbs.

These cleverly layered categories make Connections both a mental workout and a rewarding game for those who enjoy intricate puzzles that reward close attention to language and lateral thinking!

You are a competitive Connections Player that solves connections puzzles.

In Connections, the four colors—green, yellow, blue, and purple—represent the difficulty level of each category, with each color indicating how challenging the group is to identify. Here’s a breakdown of what each color typically represents and some strategic tips for solving these categories:

1. Green Category (Easiest)
Description: The green category is usually the most straightforward and often consists of easily recognizable groups, like fruits, days of the week, or colors. The connections here are usually concrete and widely recognized.
Strategy: Look for obvious, familiar associations right away. Identifying the green group first can narrow down possible matches and provide an initial foothold in solving the puzzle.

2. Yellow Category (Moderate)
Description: The yellow category is a bit trickier and often includes more subtle connections. This category may involve common themes that are still identifiable but require a second look, such as types of vehicles or clothing items.
Strategy: After finding the green category, try to spot another obvious group that may have just a bit of complexity. Be careful, as words in the yellow category sometimes appear similar to those in green or blue groups, leading to potential overlaps.

3. Blue Category (Tricky)
Description: The blue category introduces more complexity, often containing words with dual meanings or requiring contextual understanding, such as metals used in technology or words associated with cold weather. Connections here can be thematic or based on more obscure associations.
Strategy: For the blue category, try thinking outside the box and consider alternate meanings or contexts. Be ready to test hypotheses; if a group seems possible but not obvious, it might belong to blue. Look for patterns that aren’t immediately visible but make sense when you think more broadly.

4. Purple Category (Most Challenging)
Description: The purple category is the hardest, often using clever wordplay, homophones, or idiomatic associations. These groups are designed to mislead, such as words that can be verbs or nouns or phrases that have cultural or idiomatic significance.
Strategy: For the purple category, it’s essential to keep an open mind and consider indirect relationships. Try to think abstractly—consider homophones, metaphorical meanings, or phrases. Leave the purple group until last if possible, as process of elimination with the easier groups may reveal the purple connections indirectly.

General Strategies:

Start with Obvious Pairs: Pairing two words that seem obviously related helps narrow down possibilities. For example, if apple and banana feel like they go together, check for other fruits, which might clarify the green or yellow category.
Use Process of Elimination: Identifying the easier categories first narrows down the pool of remaining words, making it easier to spot connections among the leftover, often trickier words.
Watch Out for Misdirection: Many words might seem to belong to one category but are deliberately placed to throw you off. If you find words that almost fit but seem slightly out of place, set them aside—they may belong to a harder category.

Based on the data in the purple category of Connections, here are criteria that could be used to evaluate possible groups for the purple category:

Criteria for Purple Category:
Dual Meanings or Homophones: Words that sound alike but have different meanings or words that resemble other common terms. For example, "gnu" and "knew" are homophones.

Associations through Specific Contexts: Connections that depend on recognizing thematic, cultural, or idiomatic associations. For instance, “Clue Characters” requires knowledge of the board game Clue.

Play on Words: Categories that rely on puns, alternate meanings, or idiomatic expressions (e.g., "Jack ___" with terms like “Jack Knife” and “Jack Rabbit”).

Abstract or Indirect Relationships: Connections based on metaphorical or less obvious traits, such as “Things That Are Cast” where items like “Vote” and “Shadow” are grouped by the act of casting.

Conceptual or Symbolic Links: Words that are conceptually linked rather than by strict definition, such as “Things with Keys” (like “Cryptography” and “Florida”) where “key” is used metaphorically.

Cultural or Pop Culture References: Categories that involve figures from media or well-known cultural groups, like “Spice Girls” or “Video Game Characters.”

Special Linguistic Patterns: Words sharing unique phonetic or grammatical properties, such as “Palindrome” words or “Words with Silent Letters.”

Group Members: Categories with members of specific sets, like “Members of a Trio” (e.g., "Amigo," "Tenor").

Altered Contextual Meaning: Words that share a meaning only in a specific altered context or usage, like "Words Spelled with Roman Numerals" (e.g., "MIX" or "DILL").

Lexical Tricks (such as Anagrams or Roman Numerals): Categories involving words that contain specific letter arrangements, like “Anagrams” (e.g., "Slate" and "Tesla").

These criteria reflect the creative and lateral thinking required to uncover purple connections, which are often subtle and designed to misdirect players with multiple potential interpretations.

The blue categories in Connections puzzles often involve connections that are moderately challenging, relying on recognizable but sometimes subtle associations. Here’s a set of criteria to help identify blue categories based on the provided data:

Criteria for Blue Category:
Common Functional Groupings: These are practical or functional groups that often appear in daily life, like “Kitchen Appliances,” “Cleaning Verbs,” or “Parts of a Book.” They require general knowledge but aren’t overly abstract.

Synonyms and Related Terms: Words that share similar meanings or represent variations of the same concept, such as “Synonyms for Sad” or “Slang for Money.” This often tests vocabulary and contextual understanding.

Category-Specific Terms: Groups that rely on domain-specific language, like “Rappers,” “Types of Magazines,” or “Keyboard Keys.” Recognizing these often depends on familiarity with the specific field or context.

Descriptive Qualities and Attributes: These categories group items by similar traits, such as “Shades of Green” or “Slow Animals.” They’re recognizable through adjectives or distinct characteristics.

Objects with a Shared Purpose: Words that have a unifying purpose or use, such as “Fishing Gear,” “Detergents,” or “Sunglasses.” These are typically concrete and functional connections.

Species or Types Within a Broader Category: Examples include “Birds,” “Dog Breeds,” and “Vegetables.” Recognizing these groups requires some knowledge of taxonomy or basic categorization within common categories.

Verbs with Similar Actions: These categories connect through action verbs, such as “Run Quickly” or “Words for Bother.” Identifying these often involves thinking about synonyms or related actions.

Associated by Color or Shape: Examples include “Things That Are Red” or “Toruses” (doughnut shapes). These groupings are intuitive but require thinking visually about common colors, shapes, or forms.

Familiar Pop Culture or Media References: Categories like “Disney Ducks” or “Title TV Doctors” require recognition of well-known media references, which is fairly straightforward for those familiar with pop culture.

Metaphorical or Figurative Associations: Some blue categories, like “Things That Have Keys” (piano, computer) or “Animals with Tusks,” require an understanding of items that share a metaphorical trait or part. This criterion involves some lateral thinking without deep abstraction.

These criteria make the blue categories moderately challenging and engaging, as they require players to apply general knowledge, recognize common themes, and sometimes think in terms of descriptive or functional characteristics without delving into highly abstract connections.

Criteria for Yellow Category:

Distinct Theme: Each group focuses on a clear, specific theme that unites the words in an obvious way. For example, in “Bird Sounds,” words like “Caw,” “Chirp,” and “Tweet” all relate to sounds birds make.

Common Usage Context: The words are commonly understood within the context of the category, making the theme recognizable. For instance, “Wet Weather” includes “Rain,” “Hail,” “Sleet,” and “Snow,” which are universally recognized as weather types.

Non-Redundant Members: Each word represents a distinct element of the theme without repetition. In “Modes of Transportation,” words like “Boat,” “Plane,” “Train,” and “Car” all represent unique transportation methods.

Category Exclusivity: Each word should fit specifically within the assigned category and not be easily applicable to other categories. For example, “Sneaker Brands” like “Nike,” “Adidas,” “Puma,” and “Reebok” belong exclusively to footwear brands.

Semantic Variety: Words within a category should cover a range of meanings or forms, even as they connect to the same theme. In “Synonyms for Angry,” words like “Irate,” “Livid,” “Furious,” and “Incensed” capture various intensities of anger.

Cultural Relevance: The categories should reflect familiar and culturally recognizable ideas for clarity across audiences. For example, “Streaming Services” like “Netflix,” “Prime,” “Peacock,” and “Hulu” are popular streaming platforms.

Generalizability: Categories should be broad enough to encompass universally known ideas, rather than being too niche. For example, “States of Matter” with words like “Solid,” “Liquid,” “Gas,” and “Plasma” is easily understood without specialized knowledge.

These criteria ensure that the yellow category maintains consistency and clarity, with connections that are straightforward and readily understandable by a wide audience.

This color-coded system in Connections creates a layered challenge, making each puzzle a satisfying mix of immediate recognition and deeper, more nuanced thinking. By balancing familiar and tricky groupings, the game encourages strategic thought and a flexible approach!

You are a competitive Connections Player that solves connections puzzles. 

The Green Row of the puzzel is already solved. You are given 2 lists of words. The first list of words contains 12 words. The second list of words
are 4 words and represents the words that are solved correctly by the green row.

You need to give a list of 4 words that are not in the first list of words that you are confident is another row.\n\n"""

    prompt2 = r"""Connections is a clever, engaging puzzle game that challenges players to group words based on hidden relationships. In each round, players are presented with a grid of 16 words and must identify four unique categories, with each category grouping four words that share a common theme or concept. The challenge lies in spotting subtle associations, distinguishing close but incorrect connections, and using deductive reasoning to avoid errors.

The game tests and sharpens vocabulary, lateral thinking, and pattern recognition, offering a refreshing blend of strategy and wordplay. Each level becomes progressively more complex, making Connections perfect for players who enjoy thought-provoking puzzles and honing their logical skills.

The categories in Connections are often cleverly crafted to challenge players' linguistic intuition and reasoning skills. Here’s how they use subtlety, misdirection, and semantic complexity to elevate the challenge:

Homophones and Similar Sounding Words: Some categories include words that sound alike but have different meanings or spellings (e.g., pair vs. pear vs. pare). This forces players to think beyond the spelling and consider phonetic relationships, often leading to unexpected connections.

Multi-Layered Meanings: Certain words can belong to more than one category based on their different meanings or uses. For example, apple might fit into a category about fruit but could also relate to tech companies. This creates an ambiguity that requires players to think critically about context.

Thematic Misleading Associations: Some categories are intentionally deceptive by including words that seem related but don't quite fit the actual group. For example, lemon, lime, blue, and berry could trick players into thinking it’s about fruits when it’s actually colors.

Hidden Relationships and Conceptual Links: Categories sometimes rely on less obvious, abstract connections, like grouping words that represent emotions, philosophical concepts, or actions (e.g., hope, faith, love, charity). These categories require players to recognize more nuanced themes rather than concrete groups.

Cultural or Idiomatic Associations: Categories may include words that only connect in certain cultural or idiomatic contexts, such as phrases or common expressions (e.g., cold, turkey, wild, goose for "wild goose chase" and "cold turkey"). Recognizing these requires a certain familiarity with idioms or pop culture references.

Linguistic Tricks with Parts of Speech: Words may be grouped based on their grammatical function (like verbs, nouns, or adjectives), even if they have no thematic connection otherwise. For instance, words like run, jump, fly, and dive could lead players to a "sports" misinterpretation when it’s actually about actions or verbs.

These cleverly layered categories make Connections both a mental workout and a rewarding game for those who enjoy intricate puzzles that reward close attention to language and lateral thinking!

You are a competitive Connections Player that solves connections puzzles.

In Connections, the four colors—green, yellow, blue, and purple—represent the difficulty level of each category, with each color indicating how challenging the group is to identify. Here’s a breakdown of what each color typically represents and some strategic tips for solving these categories:

1. Green Category (Easiest)
Description: The green category is usually the most straightforward and often consists of easily recognizable groups, like fruits, days of the week, or colors. The connections here are usually concrete and widely recognized.
Strategy: Look for obvious, familiar associations right away. Identifying the green group first can narrow down possible matches and provide an initial foothold in solving the puzzle.

2. Yellow Category (Moderate)
Description: The yellow category is a bit trickier and often includes more subtle connections. This category may involve common themes that are still identifiable but require a second look, such as types of vehicles or clothing items.
Strategy: After finding the green category, try to spot another obvious group that may have just a bit of complexity. Be careful, as words in the yellow category sometimes appear similar to those in green or blue groups, leading to potential overlaps.

3. Blue Category (Tricky)
Description: The blue category introduces more complexity, often containing words with dual meanings or requiring contextual understanding, such as metals used in technology or words associated with cold weather. Connections here can be thematic or based on more obscure associations.
Strategy: For the blue category, try thinking outside the box and consider alternate meanings or contexts. Be ready to test hypotheses; if a group seems possible but not obvious, it might belong to blue. Look for patterns that aren’t immediately visible but make sense when you think more broadly.

4. Purple Category (Most Challenging)
Description: The purple category is the hardest, often using clever wordplay, homophones, or idiomatic associations. These groups are designed to mislead, such as words that can be verbs or nouns or phrases that have cultural or idiomatic significance.
Strategy: For the purple category, it’s essential to keep an open mind and consider indirect relationships. Try to think abstractly—consider homophones, metaphorical meanings, or phrases. Leave the purple group until last if possible, as process of elimination with the easier groups may reveal the purple connections indirectly.

General Strategies:

Start with Obvious Pairs: Pairing two words that seem obviously related helps narrow down possibilities. For example, if apple and banana feel like they go together, check for other fruits, which might clarify the green or yellow category.
Use Process of Elimination: Identifying the easier categories first narrows down the pool of remaining words, making it easier to spot connections among the leftover, often trickier words.
Watch Out for Misdirection: Many words might seem to belong to one category but are deliberately placed to throw you off. If you find words that almost fit but seem slightly out of place, set them aside—they may belong to a harder category.

Based on the data in the purple category of Connections, here are criteria that could be used to evaluate possible groups for the purple category:

Criteria for Purple Category:
Dual Meanings or Homophones: Words that sound alike but have different meanings or words that resemble other common terms. For example, "gnu" and "knew" are homophones.

Associations through Specific Contexts: Connections that depend on recognizing thematic, cultural, or idiomatic associations. For instance, “Clue Characters” requires knowledge of the board game Clue.

Play on Words: Categories that rely on puns, alternate meanings, or idiomatic expressions (e.g., "Jack ___" with terms like “Jack Knife” and “Jack Rabbit”).

Abstract or Indirect Relationships: Connections based on metaphorical or less obvious traits, such as “Things That Are Cast” where items like “Vote” and “Shadow” are grouped by the act of casting.

Conceptual or Symbolic Links: Words that are conceptually linked rather than by strict definition, such as “Things with Keys” (like “Cryptography” and “Florida”) where “key” is used metaphorically.

Cultural or Pop Culture References: Categories that involve figures from media or well-known cultural groups, like “Spice Girls” or “Video Game Characters.”

Special Linguistic Patterns: Words sharing unique phonetic or grammatical properties, such as “Palindrome” words or “Words with Silent Letters.”

Group Members: Categories with members of specific sets, like “Members of a Trio” (e.g., "Amigo," "Tenor").

Altered Contextual Meaning: Words that share a meaning only in a specific altered context or usage, like "Words Spelled with Roman Numerals" (e.g., "MIX" or "DILL").

Lexical Tricks (such as Anagrams or Roman Numerals): Categories involving words that contain specific letter arrangements, like “Anagrams” (e.g., "Slate" and "Tesla").

These criteria reflect the creative and lateral thinking required to uncover purple connections, which are often subtle and designed to misdirect players with multiple potential interpretations.

The blue categories in Connections puzzles often involve connections that are moderately challenging, relying on recognizable but sometimes subtle associations. Here’s a set of criteria to help identify blue categories based on the provided data:

Criteria for Blue Category:
Common Functional Groupings: These are practical or functional groups that often appear in daily life, like “Kitchen Appliances,” “Cleaning Verbs,” or “Parts of a Book.” They require general knowledge but aren’t overly abstract.

Synonyms and Related Terms: Words that share similar meanings or represent variations of the same concept, such as “Synonyms for Sad” or “Slang for Money.” This often tests vocabulary and contextual understanding.

Category-Specific Terms: Groups that rely on domain-specific language, like “Rappers,” “Types of Magazines,” or “Keyboard Keys.” Recognizing these often depends on familiarity with the specific field or context.

Descriptive Qualities and Attributes: These categories group items by similar traits, such as “Shades of Green” or “Slow Animals.” They’re recognizable through adjectives or distinct characteristics.

Objects with a Shared Purpose: Words that have a unifying purpose or use, such as “Fishing Gear,” “Detergents,” or “Sunglasses.” These are typically concrete and functional connections.

Species or Types Within a Broader Category: Examples include “Birds,” “Dog Breeds,” and “Vegetables.” Recognizing these groups requires some knowledge of taxonomy or basic categorization within common categories.

Verbs with Similar Actions: These categories connect through action verbs, such as “Run Quickly” or “Words for Bother.” Identifying these often involves thinking about synonyms or related actions.

Associated by Color or Shape: Examples include “Things That Are Red” or “Toruses” (doughnut shapes). These groupings are intuitive but require thinking visually about common colors, shapes, or forms.

Familiar Pop Culture or Media References: Categories like “Disney Ducks” or “Title TV Doctors” require recognition of well-known media references, which is fairly straightforward for those familiar with pop culture.

Metaphorical or Figurative Associations: Some blue categories, like “Things That Have Keys” (piano, computer) or “Animals with Tusks,” require an understanding of items that share a metaphorical trait or part. This criterion involves some lateral thinking without deep abstraction.

These criteria make the blue categories moderately challenging and engaging, as they require players to apply general knowledge, recognize common themes, and sometimes think in terms of descriptive or functional characteristics without delving into highly abstract connections.

Criteria for Yellow Category:

Distinct Theme: Each group focuses on a clear, specific theme that unites the words in an obvious way. For example, in “Bird Sounds,” words like “Caw,” “Chirp,” and “Tweet” all relate to sounds birds make.

Common Usage Context: The words are commonly understood within the context of the category, making the theme recognizable. For instance, “Wet Weather” includes “Rain,” “Hail,” “Sleet,” and “Snow,” which are universally recognized as weather types.

Non-Redundant Members: Each word represents a distinct element of the theme without repetition. In “Modes of Transportation,” words like “Boat,” “Plane,” “Train,” and “Car” all represent unique transportation methods.

Category Exclusivity: Each word should fit specifically within the assigned category and not be easily applicable to other categories. For example, “Sneaker Brands” like “Nike,” “Adidas,” “Puma,” and “Reebok” belong exclusively to footwear brands.

Semantic Variety: Words within a category should cover a range of meanings or forms, even as they connect to the same theme. In “Synonyms for Angry,” words like “Irate,” “Livid,” “Furious,” and “Incensed” capture various intensities of anger.

Cultural Relevance: The categories should reflect familiar and culturally recognizable ideas for clarity across audiences. For example, “Streaming Services” like “Netflix,” “Prime,” “Peacock,” and “Hulu” are popular streaming platforms.

Generalizability: Categories should be broad enough to encompass universally known ideas, rather than being too niche. For example, “States of Matter” with words like “Solid,” “Liquid,” “Gas,” and “Plasma” is easily understood without specialized knowledge.

These criteria ensure that the yellow category maintains consistency and clarity, with connections that are straightforward and readily understandable by a wide audience.

This color-coded system in Connections creates a layered challenge, making each puzzle a satisfying mix of immediate recognition and deeper, more nuanced thinking. By balancing familiar and tricky groupings, the game encourages strategic thought and a flexible approach!

You are a competitive Connections Coach that is helping players improve their puzzle-solving skills. Give some feedback on the following thinking and logic process\n\n"""

    prompt3 = r"""Connections is a clever, engaging puzzle game that challenges players to group words based on hidden relationships. In each round, players are presented with a grid of 16 words and must identify four unique categories, with each category grouping four words that share a common theme or concept. The challenge lies in spotting subtle associations, distinguishing close but incorrect connections, and using deductive reasoning to avoid errors.

The game tests and sharpens vocabulary, lateral thinking, and pattern recognition, offering a refreshing blend of strategy and wordplay. Each level becomes progressively more complex, making Connections perfect for players who enjoy thought-provoking puzzles and honing their logical skills.

The categories in Connections are often cleverly crafted to challenge players' linguistic intuition and reasoning skills. Here’s how they use subtlety, misdirection, and semantic complexity to elevate the challenge:

Homophones and Similar Sounding Words: Some categories include words that sound alike but have different meanings or spellings (e.g., pair vs. pear vs. pare). This forces players to think beyond the spelling and consider phonetic relationships, often leading to unexpected connections.

Multi-Layered Meanings: Certain words can belong to more than one category based on their different meanings or uses. For example, apple might fit into a category about fruit but could also relate to tech companies. This creates an ambiguity that requires players to think critically about context.

Thematic Misleading Associations: Some categories are intentionally deceptive by including words that seem related but don't quite fit the actual group. For example, lemon, lime, blue, and berry could trick players into thinking it’s about fruits when it’s actually colors.

Hidden Relationships and Conceptual Links: Categories sometimes rely on less obvious, abstract connections, like grouping words that represent emotions, philosophical concepts, or actions (e.g., hope, faith, love, charity). These categories require players to recognize more nuanced themes rather than concrete groups.

Cultural or Idiomatic Associations: Categories may include words that only connect in certain cultural or idiomatic contexts, such as phrases or common expressions (e.g., cold, turkey, wild, goose for "wild goose chase" and "cold turkey"). Recognizing these requires a certain familiarity with idioms or pop culture references.

Linguistic Tricks with Parts of Speech: Words may be grouped based on their grammatical function (like verbs, nouns, or adjectives), even if they have no thematic connection otherwise. For instance, words like run, jump, fly, and dive could lead players to a "sports" misinterpretation when it’s actually about actions or verbs.

These cleverly layered categories make Connections both a mental workout and a rewarding game for those who enjoy intricate puzzles that reward close attention to language and lateral thinking!

You are a competitive Connections Player that solves connections puzzles.

In Connections, the four colors—green, yellow, blue, and purple—represent the difficulty level of each category, with each color indicating how challenging the group is to identify. Here’s a breakdown of what each color typically represents and some strategic tips for solving these categories:

1. Green Category (Easiest)
Description: The green category is usually the most straightforward and often consists of easily recognizable groups, like fruits, days of the week, or colors. The connections here are usually concrete and widely recognized.
Strategy: Look for obvious, familiar associations right away. Identifying the green group first can narrow down possible matches and provide an initial foothold in solving the puzzle.

2. Yellow Category (Moderate)
Description: The yellow category is a bit trickier and often includes more subtle connections. This category may involve common themes that are still identifiable but require a second look, such as types of vehicles or clothing items.
Strategy: After finding the green category, try to spot another obvious group that may have just a bit of complexity. Be careful, as words in the yellow category sometimes appear similar to those in green or blue groups, leading to potential overlaps.

3. Blue Category (Tricky)
Description: The blue category introduces more complexity, often containing words with dual meanings or requiring contextual understanding, such as metals used in technology or words associated with cold weather. Connections here can be thematic or based on more obscure associations.
Strategy: For the blue category, try thinking outside the box and consider alternate meanings or contexts. Be ready to test hypotheses; if a group seems possible but not obvious, it might belong to blue. Look for patterns that aren’t immediately visible but make sense when you think more broadly.

4. Purple Category (Most Challenging)
Description: The purple category is the hardest, often using clever wordplay, homophones, or idiomatic associations. These groups are designed to mislead, such as words that can be verbs or nouns or phrases that have cultural or idiomatic significance.
Strategy: For the purple category, it’s essential to keep an open mind and consider indirect relationships. Try to think abstractly—consider homophones, metaphorical meanings, or phrases. Leave the purple group until last if possible, as process of elimination with the easier groups may reveal the purple connections indirectly.

General Strategies:

Start with Obvious Pairs: Pairing two words that seem obviously related helps narrow down possibilities. For example, if apple and banana feel like they go together, check for other fruits, which might clarify the green or yellow category.
Use Process of Elimination: Identifying the easier categories first narrows down the pool of remaining words, making it easier to spot connections among the leftover, often trickier words.
Watch Out for Misdirection: Many words might seem to belong to one category but are deliberately placed to throw you off. If you find words that almost fit but seem slightly out of place, set them aside—they may belong to a harder category.

Based on the data in the purple category of Connections, here are criteria that could be used to evaluate possible groups for the purple category:

Criteria for Purple Category:
Dual Meanings or Homophones: Words that sound alike but have different meanings or words that resemble other common terms. For example, "gnu" and "knew" are homophones.

Associations through Specific Contexts: Connections that depend on recognizing thematic, cultural, or idiomatic associations. For instance, “Clue Characters” requires knowledge of the board game Clue.

Play on Words: Categories that rely on puns, alternate meanings, or idiomatic expressions (e.g., "Jack ___" with terms like “Jack Knife” and “Jack Rabbit”).

Abstract or Indirect Relationships: Connections based on metaphorical or less obvious traits, such as “Things That Are Cast” where items like “Vote” and “Shadow” are grouped by the act of casting.

Conceptual or Symbolic Links: Words that are conceptually linked rather than by strict definition, such as “Things with Keys” (like “Cryptography” and “Florida”) where “key” is used metaphorically.

Cultural or Pop Culture References: Categories that involve figures from media or well-known cultural groups, like “Spice Girls” or “Video Game Characters.”

Special Linguistic Patterns: Words sharing unique phonetic or grammatical properties, such as “Palindrome” words or “Words with Silent Letters.”

Group Members: Categories with members of specific sets, like “Members of a Trio” (e.g., "Amigo," "Tenor").

Altered Contextual Meaning: Words that share a meaning only in a specific altered context or usage, like "Words Spelled with Roman Numerals" (e.g., "MIX" or "DILL").

Lexical Tricks (such as Anagrams or Roman Numerals): Categories involving words that contain specific letter arrangements, like “Anagrams” (e.g., "Slate" and "Tesla").

These criteria reflect the creative and lateral thinking required to uncover purple connections, which are often subtle and designed to misdirect players with multiple potential interpretations.

The blue categories in Connections puzzles often involve connections that are moderately challenging, relying on recognizable but sometimes subtle associations. Here’s a set of criteria to help identify blue categories based on the provided data:

Criteria for Blue Category:
Common Functional Groupings: These are practical or functional groups that often appear in daily life, like “Kitchen Appliances,” “Cleaning Verbs,” or “Parts of a Book.” They require general knowledge but aren’t overly abstract.

Synonyms and Related Terms: Words that share similar meanings or represent variations of the same concept, such as “Synonyms for Sad” or “Slang for Money.” This often tests vocabulary and contextual understanding.

Category-Specific Terms: Groups that rely on domain-specific language, like “Rappers,” “Types of Magazines,” or “Keyboard Keys.” Recognizing these often depends on familiarity with the specific field or context.

Descriptive Qualities and Attributes: These categories group items by similar traits, such as “Shades of Green” or “Slow Animals.” They’re recognizable through adjectives or distinct characteristics.

Objects with a Shared Purpose: Words that have a unifying purpose or use, such as “Fishing Gear,” “Detergents,” or “Sunglasses.” These are typically concrete and functional connections.

Species or Types Within a Broader Category: Examples include “Birds,” “Dog Breeds,” and “Vegetables.” Recognizing these groups requires some knowledge of taxonomy or basic categorization within common categories.

Verbs with Similar Actions: These categories connect through action verbs, such as “Run Quickly” or “Words for Bother.” Identifying these often involves thinking about synonyms or related actions.

Associated by Color or Shape: Examples include “Things That Are Red” or “Toruses” (doughnut shapes). These groupings are intuitive but require thinking visually about common colors, shapes, or forms.

Familiar Pop Culture or Media References: Categories like “Disney Ducks” or “Title TV Doctors” require recognition of well-known media references, which is fairly straightforward for those familiar with pop culture.

Metaphorical or Figurative Associations: Some blue categories, like “Things That Have Keys” (piano, computer) or “Animals with Tusks,” require an understanding of items that share a metaphorical trait or part. This criterion involves some lateral thinking without deep abstraction.

These criteria make the blue categories moderately challenging and engaging, as they require players to apply general knowledge, recognize common themes, and sometimes think in terms of descriptive or functional characteristics without delving into highly abstract connections.

Criteria for Yellow Category:

Distinct Theme: Each group focuses on a clear, specific theme that unites the words in an obvious way. For example, in “Bird Sounds,” words like “Caw,” “Chirp,” and “Tweet” all relate to sounds birds make.

Common Usage Context: The words are commonly understood within the context of the category, making the theme recognizable. For instance, “Wet Weather” includes “Rain,” “Hail,” “Sleet,” and “Snow,” which are universally recognized as weather types.

Non-Redundant Members: Each word represents a distinct element of the theme without repetition. In “Modes of Transportation,” words like “Boat,” “Plane,” “Train,” and “Car” all represent unique transportation methods.

Category Exclusivity: Each word should fit specifically within the assigned category and not be easily applicable to other categories. For example, “Sneaker Brands” like “Nike,” “Adidas,” “Puma,” and “Reebok” belong exclusively to footwear brands.

Semantic Variety: Words within a category should cover a range of meanings or forms, even as they connect to the same theme. In “Synonyms for Angry,” words like “Irate,” “Livid,” “Furious,” and “Incensed” capture various intensities of anger.

Cultural Relevance: The categories should reflect familiar and culturally recognizable ideas for clarity across audiences. For example, “Streaming Services” like “Netflix,” “Prime,” “Peacock,” and “Hulu” are popular streaming platforms.

Generalizability: Categories should be broad enough to encompass universally known ideas, rather than being too niche. For example, “States of Matter” with words like “Solid,” “Liquid,” “Gas,” and “Plasma” is easily understood without specialized knowledge.

These criteria ensure that the yellow category maintains consistency and clarity, with connections that are straightforward and readily understandable by a wide audience.

This color-coded system in Connections creates a layered challenge, making each puzzle a satisfying mix of immediate recognition and deeper, more nuanced thinking. By balancing familiar and tricky groupings, the game encourages strategic thought and a flexible approach!

You are a competitive Connections Player that solves connections puzzles. 

The Green and Yellow Rows of the puzzel are already solved. You are given 2 lists of words. The first list of words contains 8 words. The second list of words
are 8 words and represents the words that are solved correctly by the green row and yellow rows.

You need to give a list of 4 words that are not in the first list of words that you are confident is another row.\n\n"""

    prompt4 = r"""Connections is a clever, engaging puzzle game that challenges players to group words based on hidden relationships. In each round, players are presented with a grid of 16 words and must identify four unique categories, with each category grouping four words that share a common theme or concept. The challenge lies in spotting subtle associations, distinguishing close but incorrect connections, and using deductive reasoning to avoid errors.

The game tests and sharpens vocabulary, lateral thinking, and pattern recognition, offering a refreshing blend of strategy and wordplay. Each level becomes progressively more complex, making Connections perfect for players who enjoy thought-provoking puzzles and honing their logical skills.

The categories in Connections are often cleverly crafted to challenge players' linguistic intuition and reasoning skills. Here’s how they use subtlety, misdirection, and semantic complexity to elevate the challenge:

Homophones and Similar Sounding Words: Some categories include words that sound alike but have different meanings or spellings (e.g., pair vs. pear vs. pare). This forces players to think beyond the spelling and consider phonetic relationships, often leading to unexpected connections.

Multi-Layered Meanings: Certain words can belong to more than one category based on their different meanings or uses. For example, apple might fit into a category about fruit but could also relate to tech companies. This creates an ambiguity that requires players to think critically about context.

Thematic Misleading Associations: Some categories are intentionally deceptive by including words that seem related but don't quite fit the actual group. For example, lemon, lime, blue, and berry could trick players into thinking it’s about fruits when it’s actually colors.

Hidden Relationships and Conceptual Links: Categories sometimes rely on less obvious, abstract connections, like grouping words that represent emotions, philosophical concepts, or actions (e.g., hope, faith, love, charity). These categories require players to recognize more nuanced themes rather than concrete groups.

Cultural or Idiomatic Associations: Categories may include words that only connect in certain cultural or idiomatic contexts, such as phrases or common expressions (e.g., cold, turkey, wild, goose for "wild goose chase" and "cold turkey"). Recognizing these requires a certain familiarity with idioms or pop culture references.

Linguistic Tricks with Parts of Speech: Words may be grouped based on their grammatical function (like verbs, nouns, or adjectives), even if they have no thematic connection otherwise. For instance, words like run, jump, fly, and dive could lead players to a "sports" misinterpretation when it’s actually about actions or verbs.

These cleverly layered categories make Connections both a mental workout and a rewarding game for those who enjoy intricate puzzles that reward close attention to language and lateral thinking!

You are a competitive Connections Player that solves connections puzzles.

In Connections, the four colors—green, yellow, blue, and purple—represent the difficulty level of each category, with each color indicating how challenging the group is to identify. Here’s a breakdown of what each color typically represents and some strategic tips for solving these categories:

1. Green Category (Easiest)
Description: The green category is usually the most straightforward and often consists of easily recognizable groups, like fruits, days of the week, or colors. The connections here are usually concrete and widely recognized.
Strategy: Look for obvious, familiar associations right away. Identifying the green group first can narrow down possible matches and provide an initial foothold in solving the puzzle.

2. Yellow Category (Moderate)
Description: The yellow category is a bit trickier and often includes more subtle connections. This category may involve common themes that are still identifiable but require a second look, such as types of vehicles or clothing items.
Strategy: After finding the green category, try to spot another obvious group that may have just a bit of complexity. Be careful, as words in the yellow category sometimes appear similar to those in green or blue groups, leading to potential overlaps.

3. Blue Category (Tricky)
Description: The blue category introduces more complexity, often containing words with dual meanings or requiring contextual understanding, such as metals used in technology or words associated with cold weather. Connections here can be thematic or based on more obscure associations.
Strategy: For the blue category, try thinking outside the box and consider alternate meanings or contexts. Be ready to test hypotheses; if a group seems possible but not obvious, it might belong to blue. Look for patterns that aren’t immediately visible but make sense when you think more broadly.

4. Purple Category (Most Challenging)
Description: The purple category is the hardest, often using clever wordplay, homophones, or idiomatic associations. These groups are designed to mislead, such as words that can be verbs or nouns or phrases that have cultural or idiomatic significance.
Strategy: For the purple category, it’s essential to keep an open mind and consider indirect relationships. Try to think abstractly—consider homophones, metaphorical meanings, or phrases. Leave the purple group until last if possible, as process of elimination with the easier groups may reveal the purple connections indirectly.

General Strategies:

Start with Obvious Pairs: Pairing two words that seem obviously related helps narrow down possibilities. For example, if apple and banana feel like they go together, check for other fruits, which might clarify the green or yellow category.
Use Process of Elimination: Identifying the easier categories first narrows down the pool of remaining words, making it easier to spot connections among the leftover, often trickier words.
Watch Out for Misdirection: Many words might seem to belong to one category but are deliberately placed to throw you off. If you find words that almost fit but seem slightly out of place, set them aside—they may belong to a harder category.

Based on the data in the purple category of Connections, here are criteria that could be used to evaluate possible groups for the purple category:

Criteria for Purple Category:
Dual Meanings or Homophones: Words that sound alike but have different meanings or words that resemble other common terms. For example, "gnu" and "knew" are homophones.

Associations through Specific Contexts: Connections that depend on recognizing thematic, cultural, or idiomatic associations. For instance, “Clue Characters” requires knowledge of the board game Clue.

Play on Words: Categories that rely on puns, alternate meanings, or idiomatic expressions (e.g., "Jack ___" with terms like “Jack Knife” and “Jack Rabbit”).

Abstract or Indirect Relationships: Connections based on metaphorical or less obvious traits, such as “Things That Are Cast” where items like “Vote” and “Shadow” are grouped by the act of casting.

Conceptual or Symbolic Links: Words that are conceptually linked rather than by strict definition, such as “Things with Keys” (like “Cryptography” and “Florida”) where “key” is used metaphorically.

Cultural or Pop Culture References: Categories that involve figures from media or well-known cultural groups, like “Spice Girls” or “Video Game Characters.”

Special Linguistic Patterns: Words sharing unique phonetic or grammatical properties, such as “Palindrome” words or “Words with Silent Letters.”

Group Members: Categories with members of specific sets, like “Members of a Trio” (e.g., "Amigo," "Tenor").

Altered Contextual Meaning: Words that share a meaning only in a specific altered context or usage, like "Words Spelled with Roman Numerals" (e.g., "MIX" or "DILL").

Lexical Tricks (such as Anagrams or Roman Numerals): Categories involving words that contain specific letter arrangements, like “Anagrams” (e.g., "Slate" and "Tesla").

These criteria reflect the creative and lateral thinking required to uncover purple connections, which are often subtle and designed to misdirect players with multiple potential interpretations.

The blue categories in Connections puzzles often involve connections that are moderately challenging, relying on recognizable but sometimes subtle associations. Here’s a set of criteria to help identify blue categories based on the provided data:

Criteria for Blue Category:
Common Functional Groupings: These are practical or functional groups that often appear in daily life, like “Kitchen Appliances,” “Cleaning Verbs,” or “Parts of a Book.” They require general knowledge but aren’t overly abstract.

Synonyms and Related Terms: Words that share similar meanings or represent variations of the same concept, such as “Synonyms for Sad” or “Slang for Money.” This often tests vocabulary and contextual understanding.

Category-Specific Terms: Groups that rely on domain-specific language, like “Rappers,” “Types of Magazines,” or “Keyboard Keys.” Recognizing these often depends on familiarity with the specific field or context.

Descriptive Qualities and Attributes: These categories group items by similar traits, such as “Shades of Green” or “Slow Animals.” They’re recognizable through adjectives or distinct characteristics.

Objects with a Shared Purpose: Words that have a unifying purpose or use, such as “Fishing Gear,” “Detergents,” or “Sunglasses.” These are typically concrete and functional connections.

Species or Types Within a Broader Category: Examples include “Birds,” “Dog Breeds,” and “Vegetables.” Recognizing these groups requires some knowledge of taxonomy or basic categorization within common categories.

Verbs with Similar Actions: These categories connect through action verbs, such as “Run Quickly” or “Words for Bother.” Identifying these often involves thinking about synonyms or related actions.

Associated by Color or Shape: Examples include “Things That Are Red” or “Toruses” (doughnut shapes). These groupings are intuitive but require thinking visually about common colors, shapes, or forms.

Familiar Pop Culture or Media References: Categories like “Disney Ducks” or “Title TV Doctors” require recognition of well-known media references, which is fairly straightforward for those familiar with pop culture.

Metaphorical or Figurative Associations: Some blue categories, like “Things That Have Keys” (piano, computer) or “Animals with Tusks,” require an understanding of items that share a metaphorical trait or part. This criterion involves some lateral thinking without deep abstraction.

These criteria make the blue categories moderately challenging and engaging, as they require players to apply general knowledge, recognize common themes, and sometimes think in terms of descriptive or functional characteristics without delving into highly abstract connections.

Criteria for Yellow Category:

Distinct Theme: Each group focuses on a clear, specific theme that unites the words in an obvious way. For example, in “Bird Sounds,” words like “Caw,” “Chirp,” and “Tweet” all relate to sounds birds make.

Common Usage Context: The words are commonly understood within the context of the category, making the theme recognizable. For instance, “Wet Weather” includes “Rain,” “Hail,” “Sleet,” and “Snow,” which are universally recognized as weather types.

Non-Redundant Members: Each word represents a distinct element of the theme without repetition. In “Modes of Transportation,” words like “Boat,” “Plane,” “Train,” and “Car” all represent unique transportation methods.

Category Exclusivity: Each word should fit specifically within the assigned category and not be easily applicable to other categories. For example, “Sneaker Brands” like “Nike,” “Adidas,” “Puma,” and “Reebok” belong exclusively to footwear brands.

Semantic Variety: Words within a category should cover a range of meanings or forms, even as they connect to the same theme. In “Synonyms for Angry,” words like “Irate,” “Livid,” “Furious,” and “Incensed” capture various intensities of anger.

Cultural Relevance: The categories should reflect familiar and culturally recognizable ideas for clarity across audiences. For example, “Streaming Services” like “Netflix,” “Prime,” “Peacock,” and “Hulu” are popular streaming platforms.

Generalizability: Categories should be broad enough to encompass universally known ideas, rather than being too niche. For example, “States of Matter” with words like “Solid,” “Liquid,” “Gas,” and “Plasma” is easily understood without specialized knowledge.

These criteria ensure that the yellow category maintains consistency and clarity, with connections that are straightforward and readily understandable by a wide audience.

This color-coded system in Connections creates a layered challenge, making each puzzle a satisfying mix of immediate recognition and deeper, more nuanced thinking. By balancing familiar and tricky groupings, the game encourages strategic thought and a flexible approach!

You are a competitive Connections Coach that is helping players improve their puzzle-solving skills. Give some feedback on the following thinking and logic process\n\n"""

    prompt_ending=r"""Carefully think through this challenge and explain your answer in detail"""
    
    def get_prompt(self, words, solved_words, previous_guesses):
        return f"{self.prompt}words: {words}\nsolved words: {solved_words}\nprevious guesses: {previous_guesses}\n\n{self.prompt_ending}"
    
    def get_prompt_2(self, words, solved_words, previous_guesses, thinking):
        return f"{self.prompt2}words: {words}\nsolved words: {solved_words}\nprevious guesses: {previous_guesses}\nThinking: {thinking}\n\n{self.prompt_ending}"
    
    def get_prompt_3(self, words, solved_words, previous_guesses):
        return f"{self.prompt3}words: {words}\nsolved words: {solved_words}\nprevious guesses: {previous_guesses}\n\n{self.prompt_ending}"
    
    def get_prompt_4(self, words, solved_words, previous_guesses, thinking):
        return f"{self.prompt4}words: {words}\nsolved words: {solved_words}\nprevious guesses: {previous_guesses}\nThinking: {thinking}\n\n{self.prompt_ending}"
    
    def __init__(self):
        gemini.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = gemini.GenerativeModel('gemini-1.5-flash')
        self.llm2 = gemini.GenerativeModel('gemini-1.5-flash')
        self.chat = self.llm.start_chat()
        
    def run(self, words, solved_words, previous_guesses):
        prompt = self.get_prompt(words, solved_words, previous_guesses)
        response = self.chat.send_message(prompt)
        
        print(response)
        
        prompt_2 = self.get_prompt_2(words, solved_words, previous_guesses, response)
        feedback = self.llm.generate_content(prompt_2)
        
        print(feedback)
        
        response = self.chat.send_message(f"Now make some logical descision based on the feedback from your coach. Time is running out so be quick and give four words that you are confident are in the same row. Provide the words as a list. feedback: {feedback}",
                                          generation_config=gemini.GenerationConfig(response_mime_type="application/json"))
        
        return response
    
    def run2(self, words, solved_words, previous_guesses):
        prompt = self.get_prompt_3(words, solved_words, previous_guesses)
        response = self.chat.send_message(prompt)
        
        print(response)
        
        prompt_2 = self.get_prompt_4(words, solved_words, previous_guesses, response)
        feedback = self.llm.generate_content(prompt_2)
        
        print(feedback)
        
        response = self.chat.send_message(f"Now make some logical descision based on the feedback from your coach. Time is running out so be quick and give four words that you are confident are in the same row. Provide the words as a list. feedback: {feedback}",
                                          generation_config=gemini.GenerationConfig(response_mime_type="application/json"))
        
        return response