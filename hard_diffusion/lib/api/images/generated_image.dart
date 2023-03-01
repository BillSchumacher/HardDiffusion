import 'package:hard_diffusion/list_page.dart';
import 'package:uuid_type/uuid_type.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'dart:convert';

class GeneratedImage {
  final Uuid taskId;
  final int id;
  final String filename;
  final String? host;
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
  final String? model;
  final bool? error;

  const GeneratedImage(
      {required this.taskId,
      required this.id,
      required this.filename,
      this.host,
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
      this.model,
      this.error});

  factory GeneratedImage.fromJson(Map<String, dynamic> json) {
    var negativePrompt = json['negative_prompt'];
    var generatedAt = json['generated_at'];
    if (generatedAt != null) {
      generatedAt = DateTime.parse(generatedAt);
    }
    var model = json['model'];
    if (model == null) {
      model = 'CompVis/StableDiffusion-1.4';
    }
    return GeneratedImage(
      taskId: Uuid.parse(json['task_id']),
      id: json['id'] as int,
      filename: json['filename'] as String,
      host: json['host'],
      createdAt: DateTime.parse(json['created_at']),
      generatedAt: generatedAt,
      prompt: json['prompt'] as String,
      negativePrompt: json['negative_prompt'],
      duration: json['duration'],
      seed: json['seed'],
      guidanceScale: json['guidance_scale'] as double,
      numInferenceSteps: json['num_inference_steps'] as int,
      height: json['height'] as int,
      width: json['width'] as int,
      model: model,
      error: json['error'] as bool,
    );
  }
}

Future<ListPage<GeneratedImage>> fetchPhotos(
    http.Client client, int lastPage, int pageSize) async {
  final response = await client
      .get(Uri.parse('http://localhost:8000/images/$lastPage/$pageSize'));

  // Use the compute function to run parsePhotos in a separate isolate.
  return compute(parsePhotos, response.body);
}

// A function that converts a response body into a List<Photo>.
ListPage<GeneratedImage> parsePhotos(String responseBody) {
  final jsonResponse = jsonDecode(responseBody);
  final images = jsonResponse["images"].cast<Map<String, dynamic>>();
  final parsed_images = images.map<GeneratedImage>((json) {
    return GeneratedImage.fromJson(json);
  }).toList();
  final parsed_total_count = jsonResponse["total"] as int;
  return ListPage<GeneratedImage>(
      itemList: parsed_images, totalCount: parsed_total_count);
}