import argparse
import os
from typing import Any, Dict, List

import cv2

from services.recognition import recognize
from utils.config_loader import load_config


def _display(title: str, paths: List[str]) -> None:
    """Show a list of images sequentially."""
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize search results")
    parser.add_argument("--image", required=True, help="Query image path")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates")
    args = parser.parse_args()

    cfg = load_config()
    frames_root = cfg["storage"].get("frames_root", "")

    result = recognize(args.image, top_k=args.top_k)
    movies: List[Dict[str, Any]] = result.get("movies", [])

    if not movies:
        print("No matches available.")
        return

    if result.get("is_unknown", True):
        print("Unknown face. Showing nearest matches by movie...")
        for movie in movies:
            movie_label = movie.get("movie") or movie.get("movie_id") or "Unknown movie"
            match_label = movie.get("match_label") or movie.get("match_status", "near_match")
            print(f"Movie: {movie_label} ({match_label})")
            for character in movie.get("characters", []):
                char_id = character.get("character_id", "unknown")
                char_label = character.get("match_label") or character.get(
                    "match_status", "near_match"
                )
                print(f"  Candidate {char_id} - {char_label}")
                images: List[str] = []
                rep = character.get("rep_image")
                if isinstance(rep, dict):
                    rep_movie = rep.get("movie") or movie.get("movie")
                    frame = rep.get("frame")
                    if rep_movie and frame:
                        images.append(os.path.join(frames_root, rep_movie, frame))
                elif isinstance(rep, str):
                    images.append(rep)
                images.extend(character.get("preview_paths") or [])
                _display(f"{char_id}:{movie_label}", images)
    else:
        print("Recognized face. Showing matches by movie...")
        for movie in movies:
            movie_label = movie.get("movie") or movie.get("movie_id") or "Unknown movie"
            match_label = movie.get("match_label") or "Recognized"
            print(f"Movie: {movie_label} ({match_label})")
            for character in movie.get("characters", []):
                char_id = character.get("character_id", "unknown")
                score = character.get("score") or character.get("distance")
                char_label = character.get("match_label") or "Recognized"
                if score is not None:
                    print(f"  Character {char_id} - {char_label} (score: {score:.4f})")
                else:
                    print(f"  Character {char_id} - {char_label}")
                images: List[str] = []
                rep = character.get("rep_image")
                if isinstance(rep, dict):
                    rep_movie = rep.get("movie") or movie.get("movie")
                    frame = rep.get("frame")
                    if rep_movie and frame:
                        images.append(os.path.join(frames_root, rep_movie, frame))
                elif isinstance(rep, str):
                    images.append(rep)
                images.extend(character.get("preview_paths") or [])
                _display(f"{char_id}:{movie_label}", images)


if __name__ == "__main__":
    main()
