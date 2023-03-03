import 'dart:async';
import 'package:hard_diffusion/api/images/generated_image.dart';
import 'package:hard_diffusion/exception_indicators/empty_list_indicator.dart';
import 'package:hard_diffusion/exception_indicators/error_indicator.dart';
import 'package:hard_diffusion/main.dart';
import 'package:infinite_scroll_pagination/infinite_scroll_pagination.dart';
import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:provider/provider.dart';

/*
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
        future: fetchPhotos(http.Client(), 0, 10),
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
*/

final Image errorImage = Image.network("http://localhost:8000/error.gif");
final Image paintingImage = Image.network("http://localhost:8000/painting.gif");

class ImageDetails extends StatefulWidget {
  const ImageDetails({super.key, required this.item});

  final GeneratedImage item;
  @override
  _ImageDetailsState createState() => _ImageDetailsState(item: item);
}

class _ImageDetailsState extends State<ImageDetails> {
  _ImageDetailsState({required this.item});
  final GeneratedImage item;
  bool showEditButton = false;
  @override
  Widget build(BuildContext context) {
    var image;
    if (item.error!) {
      image = errorImage;
    } else if (item.generatedAt == null) {
      image = paintingImage;
    } else {
      image = CachedNetworkImage(
        placeholder: (context, url) => const CircularProgressIndicator(),
        imageUrl: "http://localhost:8000/${item.filename}",
        imageBuilder: (context, imageProvider) => Container(
          decoration: BoxDecoration(
            image: DecorationImage(
              image: imageProvider,
              fit: BoxFit.fill,
              //colorFilter: ColorFilter.mode(Colors.red, BlendMode.colorBurn)),
            ),
          ),
        ),
        errorWidget: (context, url, error) => errorImage,
      );
    }
    return Stack(
      children: <Widget>[
        InkWell(
            onTap: () => setState(() => showEditButton = !showEditButton),
            child: image),
        if (showEditButton) ...[
          Positioned(
            bottom: 5,
            left: 5,
            child: ElevatedButton(
              child: Text('Copy'),
              onPressed: () => {},
            ),
          ),
          Positioned(
            bottom: 5,
            right: 5,
            child: ElevatedButton(
              child: Text('Delete'),
              onPressed: () => {},
            ),
          ),
          Positioned(
            top: 5,
            left: 5,
            child: ElevatedButton(
              child: Text('Use as Input'),
              onPressed: () => {},
            ),
          ),
          Positioned(
            top: 5,
            right: 5,
            child: ElevatedButton(
              child: Text('Use settings'),
              onPressed: () => {},
            ),
          ),
        ]
      ],
    );
  }
}

class GeneratedImageListView extends StatefulWidget {
  @override
  _GeneratedImageListViewState createState() => _GeneratedImageListViewState();
}

class _GeneratedImageListViewState extends State<GeneratedImageListView> {
  static const _pageSize = 10;

  List<GeneratedImage> _generatedImageList = [];
  final PagingController<int, GeneratedImage> _pagingController =
      PagingController(firstPageKey: 0);

  Future<void> insertGeneratedImageList(List<GeneratedImage> platformList) =>
      Future.microtask(() => _generatedImageList = platformList);

  Future<List<GeneratedImage>> getGeneratedImageList() =>
      Future.microtask(() => _generatedImageList);

  @override
  void didUpdateWidget(GeneratedImageListView oldWidget) {
    // if filters changed...
    // if (oldWidget.listPreferences != widget.listPreferences) {
    //  _pagingController.refresh();
    // }
    super.didUpdateWidget(oldWidget);
  }

  @override
  void initState() {
    _pagingController.addPageRequestListener((pageKey) {
      _fetchPage(pageKey);
    });
    super.initState();
  }

  Future<void> _fetchPage(int pageKey) async {
    try {
      final newPage = await fetchPhotos(pageKey, _pageSize);
      final previouslyFetchedItemsCount =
          // 2
          _pagingController.itemList?.length ?? 0;

      final isLastPage = newPage.isLastPage(previouslyFetchedItemsCount);
      final newItems = newPage.itemList;

      if (isLastPage) {
        // 3
        _pagingController.appendLastPage(newItems);
      } else {
        final nextPageKey = pageKey + 1;
        _pagingController.appendPage(newItems, nextPageKey);
      }
    } catch (error) {
      // 4
      _pagingController.error = error;
    }
  }

  Widget getImage(item) {
    return ImageDetails(item: item);
  }

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    if (appState.needsRefresh) {
      _pagingController.refresh();
      appState.didRefresh();
    }
    return RefreshIndicator(
        onRefresh: () => Future.sync(
              () => _pagingController.refresh(),
            ),
        child: PagedGridView(
          pagingController: _pagingController,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
          ),
          builderDelegate: PagedChildBuilderDelegate<GeneratedImage>(
            itemBuilder: (context, item, index) => getImage(item),
            firstPageErrorIndicatorBuilder: (context) => ErrorIndicator(
              error: _pagingController.error,
              onTryAgain: () => _pagingController.refresh(),
            ),
            noItemsFoundIndicatorBuilder: (context) => EmptyListIndicator(),
          ),
        ));
  }

  @override
  void dispose() {
    _pagingController.dispose();
    super.dispose();
  }
}
