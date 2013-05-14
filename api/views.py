from django.http import HttpResponse
from django.conf.urls import patterns, include, url
from django.shortcuts import render, get_object_or_404
import random
import mathgutz
import json
# import os
# import pickle
import re
import time
from collections import defaultdict
import numpy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

def home(request):
    html = "<html><body>Howdy world.</body></html>"
    return HttpResponse(html)

def post(request):

    if request.method == "POST":
        input_hash = { 45: { "content": "blah whatever meta_stuff too", "label": "fun times" }, 54: { "content": request.POST['message'], "label": "fun times" } }
        awesome = train_test_clf(input_hash)
        return render(request, 'post.html', result = awesome)
    else :
        awesome = "success"
        return render(request, 'form.html', {'result': awesome })

def query(request):
    return "stuff"

def delete(request):
    return "stuff"

def update(request):
    return "stuff"

def show(request):
    return "stuff"

def clone(request):
    return "stuff"

def index(request):
    return "stuff"

def debug(request):
    return render(request, 'debug.html', {})

def debug_result(request):
    if 'q' in request.POST:
        s = unicode(request.POST['q'])
    else:
        s = unicode("")
    try:
        t = json.loads(s)
    except ValueError:
        t = {}
    results = train_test_clf(t)
    return HttpResponse(json.dumps(results), content_type="application/json")

def train_test_clf(input_hash) :
    corpus = input_hash.keys()
    if len(corpus) < 2:
        return {"error":"too little data"}
    random.shuffle(corpus)
    size = len(corpus) / 2
    train_cor, test_cor = corpus[:size], corpus[size:]
    msgs_features = {}
    for id in corpus:
        msgs_features[id] = {}
        # words = input_hash[id]["content"].split()
        # msgs_features[id]["features"] = dict([(word,1) for word in words if len(word) > 2 ])
        words = mathgutz.happy_tokenize(input_hash[id]["content"])
        msgs_features[id]["features"] = mathgutz.extract_features(words, "bow")
        msgs_features[id]["label"] = input_hash[id]["label"]
    train_data = [(msgs_features[id]["features"], msgs_features[id]["label"]) for id in train_cor]
    tval, cval = 1e-8, 13.0
    classifier = SklearnClassifier(LogisticRegression(tol=tval, penalty='l2', C=cval)).train(train_data)
    results = {}
    for id in corpus:
        results[id] = {}
        results[id]["actual_label"] = msgs_features[id]["label"]
        results[id]["predic_label"] = classifier.classify(msgs_features[id]["features"])
    return results
