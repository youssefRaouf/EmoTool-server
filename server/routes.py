# Create your views here.
from django.http import HttpResponse
from rest_framework.decorators import api_view
import json
@api_view(['POST','GET'])
def index(request):
    if request.method == "POST":
        text = json.loads(request.body)["text"]
        return HttpResponse("msg received was: "+text)
    return HttpResponse("Get request received")