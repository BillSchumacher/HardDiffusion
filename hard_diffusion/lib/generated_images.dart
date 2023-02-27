import 'package:english_words/src/word_pair.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:hard_diffusion/main.dart';

class GeneratedImages extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var favorites = appState.favorites;
    if (favorites.isEmpty) {
      return Center(
        child: Text('No favorites yet'),
      );
    }
    return Flexible(
      child: ListView(
          scrollDirection: Axis.vertical,
          shrinkWrap: false,
          children: [
            Padding(
              padding: const EdgeInsets.all(20),
              child: Text('You have ${favorites.length} favorites, generate'),
            ),
            for (var pair in favorites)
              ListTile(
                  leading: Icon(Icons.favorite),
                  title: Text(pair.asPascalCase)),
          ]),
    );
  }
}
