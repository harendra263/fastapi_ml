from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel
from typing import List

app = FastAPI()


@app.get("/")
def read_main():
    return {"message": "Hello World"}


class Article(BaseModel):
    content: str
    comments: List[str] = []

@app.post("/article/")
def analyze_article(articles: List[Article]):
    """
    Analyze an article and extract entities with ✨ spaCy ✨

    Statistical models *will* have **errors**
    * Extract entities
    * Scream comments

    """
    ents = []
    comments = []
    for article in articles:
        for comment in article.comments:
            comments.append(comment.upper())
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({"text":ent.text, "label": ent.label_})
    return {"ents":ents, "comments":comments}
