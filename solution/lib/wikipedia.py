from .embeddings.remote_embed import RemoteEmbeddingsModel
import wikipediaapi

"""
The Api for Wikipedia Querying
"""
class WikipediaApi:
    
    def __init__(self):
        self.model = RemoteEmbeddingsModel("clustering", "models/embedding-001")
        self.client = wikipediaapi.Wikipedia('Wikipedia API', 'en')
        
    def find_article(self, name):
        return self.client.page(name)
    
    def handle_query(self, query):
        article = self.find_article(query)
        
        if not article:
            return None
        
        return self.recursive_handle_depth(query, 1)
    
    def recursive_handle_depth(self, query, depth):
        if depth == 0:
            return []
        
        article = self.find_article(query)
        
        if not article:
            return []
        
        content = list()
        
        for link in article.links:
            content += self.recursive_handle_depth(link, depth - 1)
        
        return list([article.title, article.categories, article.summary, article.text]) + content