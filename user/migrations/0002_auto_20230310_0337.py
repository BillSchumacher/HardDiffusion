# Generated by Django 3.2.18 on 2023-03-10 09:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='twitter_access_token',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='user',
            name='twitter_access_token_secret',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]