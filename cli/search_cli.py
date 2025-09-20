import argparse
import joblib
import numpy as np
import os


from cli.find_actor import _decide_single_movie
from utils.config_loader import load_config
from utils.search_actor import search_actor


def main():
    # --- Load config for defaults ---
    cfg = load_config()
    pca_cfg = cfg.get("pca", {})
    storage_cfg = cfg.get("storage", {})
    search_cfg = cfg.get("search", {})
    frames_root = storage_cfg.get("frames_root", "")

    default_sim_threshold = search_cfg.get("sim_threshold", 0.5)
    default_margin_threshold = search_cfg.get("margin_threshold", 0.05)
    default_top_k = max(2, search_cfg.get("knn", 5))
    default_min_count = search_cfg.get("min_count", 0)
    default_ratio_threshold = search_cfg.get("ratio_threshold", 1.1)
    pca_model_path = storage_cfg.get("pca_model", "models/pca_model.joblib")

    parser = argparse.ArgumentParser(description="Search actor by face image")
    parser.add_argument("--image", required=True, help="Path to the query image")
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=default_sim_threshold,
        help="Minimum similarity score for a valid match (default from config)",
    )
@@ -49,71 +50,66 @@ def main():
        help="Minimum occurrence count for characters (default from config)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=default_ratio_threshold,
        help="Minimum ratio between top-1 and top-2 similarities for a valid match (default from config)",
    )
    args = parser.parse_args()
    # --- Get embedding and search function ---
    results = search_actor(args.image, return_emb=True)
    if not results or "embedding" not in results:
        print("No matching actors found.")
        return

    emb = np.array(results["embedding"]).reshape(1, -1)

    # --- Nếu PCA được bật thì transform ---
    if pca_cfg.get("enable", False) and os.path.exists(pca_model_path):
        print(f"[INFO] Applying PCA transform from {pca_model_path}")
        pca_model = joblib.load(pca_model_path)
        emb = pca_model.transform(emb)

    # --- Search bằng embedding đã được PCA ---
    top_k = max(2, args.top_k)
    matches_by_movie = results["search_func"](
        emb, top_k=top_k, min_count=args.min_count
    )

    if not matches_by_movie:
        print("No matching actors found.")
        return

    for movie_id, matches in matches_by_movie.items():
        decision = _decide_single_movie(
            matches,
            args.sim_threshold,
            args.ratio_threshold,
            args.margin_threshold,
        )
        movie_label = matches[0].get("movie") if matches else f"movie_id={movie_id}"
        if not decision["matches"]:
            print(f"{movie_label}: Unknown")
            continue

        status = "Recognized" if decision["recognized"] else "Suggestions"
        print(f"{movie_label} – {status}:")
        for res in decision["matches"]:
            char_id = res.get("character_id", "unknown")
            score = res.get("distance", 0.0)
            print(f"  Character {char_id} - Score: {score:.4f}")
            rep = res.get("rep_image", {})
            if rep:
                movie = rep.get("movie", "")
                frame = rep.get("frame", "")
                bbox = rep.get("bbox", [])
                path = (
                    os.path.join(frames_root, movie, frame)
                    if movie and frame
                    else ""
                )
                print(f"    Representative frame: {path} bbox={bbox}")


if __name__ == "__main__":
    main()