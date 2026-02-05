//! In-memory vector store implementation.
//!
//! Useful for testing and small datasets.

use super::{cosine_similarity, Document, IndexedVideo, SearchResult, VectorStore};
use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory vector store.
pub struct MemoryVectorStore {
    documents: RwLock<HashMap<String, Document>>,
}

impl MemoryVectorStore {
    /// Create a new in-memory vector store.
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn upsert(&self, doc: &Document) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        docs.insert(doc.id.to_string(), doc.clone());
        Ok(())
    }

    async fn upsert_batch(&self, docs: &[Document]) -> Result<usize> {
        let mut store = self.documents.write().unwrap();
        for doc in docs {
            store.insert(doc.id.to_string(), doc.clone());
        }
        Ok(docs.len())
    }

    async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        self.search_with_threshold(query_embedding, limit, 0.0).await
    }

    async fn search_with_threshold(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>> {
        let docs = self.documents.read().unwrap();

        let mut results: Vec<SearchResult> = docs
            .values()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .filter(|r| r.score >= min_score)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    async fn delete_by_video_id(&self, video_id: &str) -> Result<usize> {
        let mut docs = self.documents.write().unwrap();
        let initial_len = docs.len();
        docs.retain(|_, doc| doc.video_id != video_id);
        Ok(initial_len - docs.len())
    }

    async fn list_videos(&self) -> Result<Vec<IndexedVideo>> {
        let docs = self.documents.read().unwrap();

        let mut video_map: HashMap<String, IndexedVideo> = HashMap::new();

        for doc in docs.values() {
            let entry = video_map.entry(doc.video_id.clone()).or_insert_with(|| {
                IndexedVideo {
                    video_id: doc.video_id.clone(),
                    video_title: doc.video_title.clone(),
                    chunk_count: 0,
                    total_duration_seconds: 0.0,
                    indexed_at: doc.indexed_at,
                }
            });

            entry.chunk_count += 1;
            if doc.end_seconds > entry.total_duration_seconds {
                entry.total_duration_seconds = doc.end_seconds;
            }
            if doc.indexed_at > entry.indexed_at {
                entry.indexed_at = doc.indexed_at;
            }
        }

        let mut videos: Vec<IndexedVideo> = video_map.into_values().collect();
        videos.sort_by(|a, b| b.indexed_at.cmp(&a.indexed_at));

        Ok(videos)
    }

    async fn get_video(&self, video_id: &str) -> Result<Option<IndexedVideo>> {
        let videos = self.list_videos().await?;
        Ok(videos.into_iter().find(|v| v.video_id == video_id))
    }

    async fn is_video_indexed(&self, video_id: &str) -> Result<bool> {
        let docs = self.documents.read().unwrap();
        Ok(docs.values().any(|d| d.video_id == video_id))
    }

    async fn get_by_video_id(&self, video_id: &str) -> Result<Vec<Document>> {
        let docs = self.documents.read().unwrap();
        let mut result: Vec<Document> = docs
            .values()
            .filter(|d| d.video_id == video_id)
            .cloned()
            .collect();
        result.sort_by_key(|d| d.chunk_order);
        Ok(result)
    }

    async fn document_count(&self) -> Result<usize> {
        let docs = self.documents.read().unwrap();
        Ok(docs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_vector_store() {
        let store = MemoryVectorStore::new();

        let doc1 = Document::new(
            "video1".to_string(),
            "Test Video".to_string(),
            None,
            "Hello world".to_string(),
            0.0,
            30.0,
            vec![1.0, 0.0, 0.0],
            0,
            None,
        );

        let doc2 = Document::new(
            "video1".to_string(),
            "Test Video".to_string(),
            None,
            "Goodbye world".to_string(),
            30.0,
            60.0,
            vec![0.0, 1.0, 0.0],
            1,
            None,
        );

        store.upsert_batch(&[doc1, doc2]).await.unwrap();

        assert_eq!(store.document_count().await.unwrap(), 2);

        let results = store.search(&[1.0, 0.0, 0.0], 10).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);

        let videos = store.list_videos().await.unwrap();
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].chunk_count, 2);
    }
}
