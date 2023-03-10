# Generated by Django 4.1.7 on 2023-03-03 02:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("generate", "0006_generatedimage_negative_prompt"),
    ]

    operations = [
        migrations.AddField(
            model_name="generatedimage",
            name="batch_number",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="generatedimage",
            name="owner",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="generated_images",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
