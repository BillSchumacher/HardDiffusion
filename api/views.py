import json

from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.shortcuts import redirect

import tweepy
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from rest_framework_simplejwt.tokens import RefreshToken

CHANNEL_LAYER = get_channel_layer()
GROUP_SEND = async_to_sync(CHANNEL_LAYER.group_send)


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        "refresh": str(refresh),
        "access": str(refresh.access_token),
    }


callback_url = "http://localhost:8000/api/v1/oauth/twitter/callback"
oauth_consumer_key = settings.SOCIAL_AUTH_TWITTER_KEY
oauth_consumer_secret = settings.SOCIAL_AUTH_TWITTER_SECRET


def oauth_twitter(request):
    sessionId = request.GET.get("session_id")
    oauth1_user_handler = tweepy.OAuth1UserHandler(
        oauth_consumer_key,
        oauth_consumer_secret,
        callback=f"{callback_url}?session_id={sessionId}",
    )

    url = oauth1_user_handler.get_authorization_url(signin_with_twitter=True)
    return redirect(url)


def oauth_twitter_callback(request):
    session_id = request.GET.get("session_id")
    oauth_token = request.GET.get("oauth_token")
    oauth_verifier = request.GET.get("oauth_verifier")
    oauth1_user_handler = tweepy.OAuth1UserHandler(
        oauth_consumer_key, oauth_consumer_secret, callback=callback_url
    )

    oauth1_user_handler.request_token = {
        "oauth_token": oauth_token,
        "oauth_token_secret": oauth_consumer_secret,
    }
    access_token, access_token_secret = oauth1_user_handler.get_access_token(
        oauth_verifier
    )
    api = tweepy.API(oauth1_user_handler)
    # settings = api.get_settings()
    # client = tweepy.Client(
    #    consumer_key=oauth_consumer_key,
    #    consumer_secret=oauth_consumer_secret,
    #    access_token=access_token,
    #    access_token_secret=access_token_secret
    # user = client.get_me()
    creds = api.verify_credentials(include_email=True)
    user_model = get_user_model()
    user_qs = user_model.objects.filter(email=creds.email).first()
    if not user_qs:
        print("No existing user, creating...")
        user_qs = user_model(
            email=creds.email,
            username=creds.screen_name,
            twitter_access_token=access_token,
            twitter_access_token_secret=access_token_secret,
        )
    else:
        print("Existing user, updating access token")
        user_qs.twitter_access_token = access_token
        user_qs.twitter_access_token_secret = access_token_secret
    user_qs.save()
    tokens = get_tokens_for_user(user_qs)
    GROUP_SEND(
        session_id,
        {
            "type": "event_message",
            "event": "authenticated",
            "message": json.dumps(tokens),
        },
    )
    return HttpResponse(f"Thanks return to the app!")
