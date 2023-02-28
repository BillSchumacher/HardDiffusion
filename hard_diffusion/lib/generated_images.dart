import 'dart:async';
import 'dart:convert';
import 'dart:ffi';

import 'package:english_words/src/word_pair.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:hard_diffusion/main.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:uuid_type/uuid_type.dart';

Future<List<GeneratedImage>> fetchPhotos(http.Client client) async {
  final response = await client.get(Uri.parse('http://localhost:8000/images'));

  // Use the compute function to run parsePhotos in a separate isolate.
  return compute(parsePhotos, response.body);
}

// A function that converts a response body into a List<Photo>.
List<GeneratedImage> parsePhotos(String responseBody) {
  print(responseBody);
  final parsed =
      jsonDecode(responseBody)["images"].cast<Map<String, dynamic>>();

  return parsed
      .map<GeneratedImage>((json) => GeneratedImage.fromJson(json))
      .toList();
}

/*
            "images": [
                {
                    "id": image.id,
                    "task_id": image.filename[:-4],
                    "filename": image.filename,
                    "host": image.host,
                    "created_at": image.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "generated_at": image.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if image.generated_at
                    else None,
                    "prompt": image.prompt,
                    "negative_prompt": image.negative_prompt,
                    "duration": f"{image.duration:.2f} seconds"
                    if image.duration
                    else None,
                    "seed": image.seed,
                    "guidance_scale": image.guidance_scale,
                    "num_inference_steps": image.num_inference_steps,
                    "height": image.height,
                    "width": image.width,
                    "model": image.model,
                    "error": image.error,
                }
                for image in generated_images
            ]
*/
class GeneratedImage {
  final Uuid taskId;
  final int id;
  final String filename;
  final String host;
  final DateTime createdAt;
  final DateTime? generatedAt;
  final String prompt;
  final String? negativePrompt;
  final String? duration;
  final int? seed;
  final double? guidanceScale;
  final int? numInferenceSteps;
  final int height;
  final int width;
  final String model;
  final bool? error;

  const GeneratedImage(
      {required this.taskId,
      required this.id,
      required this.filename,
      required this.host,
      required this.createdAt,
      this.generatedAt,
      required this.prompt,
      this.negativePrompt,
      this.duration,
      this.seed,
      this.guidanceScale,
      this.numInferenceSteps,
      required this.height,
      required this.width,
      required this.model,
      this.error});

  factory GeneratedImage.fromJson(Map<String, dynamic> json) {
    return GeneratedImage(
      taskId: Uuid.parse(json['task_id']),
      id: json['id'] as int,
      filename: json['filename'] as String,
      host: json['host'] as String,
      createdAt: DateTime.parse(json['created_at']),
      generatedAt: DateTime.parse(json['generated_at']),
      prompt: json['prompt'] as String,
      negativePrompt: json['negative_prompt'] as String,
      duration: json['duration'] as String,
      seed: json['seed'] as int,
      guidanceScale: json['guidance_scale'] as double,
      numInferenceSteps: json['num_inference_steps'] as int,
      height: json['height'] as int,
      width: json['width'] as int,
      model: json['model'] as String,
      error: json['error'] as bool,
    );
  }
}

class GeneratedImages extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    /*
    var appState = context.watch<MyAppState>();
    var favorites = appState.favorites;
    if (favorites.isEmpty) {
      return Center(
        child: Text('No Images yet'),
      );
    }*/
    return Flexible(
      child: FutureBuilder<List<GeneratedImage>>(
        future: fetchPhotos(http.Client()),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return const Center(
              child: Text('An error has occurred!'),
            );
          } else if (snapshot.hasData) {
            return PhotosList(photos: snapshot.data!);
          } else {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }
        },
      ),
    );
  }
}

class PhotosList extends StatelessWidget {
  const PhotosList({super.key, required this.photos});

  final List<GeneratedImage> photos;

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
      ),
      itemCount: photos.length,
      itemBuilder: (context, index) {
        return Image.network("http://localhost:8000/${photos[index].filename}");
      },
    );
  }
}
