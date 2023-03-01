class ListPage<ItemType> {
  ListPage({
    required this.totalCount,
    required this.itemList,
  })  : assert(totalCount != null),
        assert(itemList != null);

  final int totalCount;
  final List<ItemType> itemList;

  bool isLastPage(int previouslyFetchedItemsCount) {
    final newItemsCount = itemList.length;
    final totalFetchedItemsCount = previouslyFetchedItemsCount + newItemsCount;
    return totalFetchedItemsCount == totalCount;
  }
}
