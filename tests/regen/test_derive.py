"""How the staged media reaches the browser: what is sent, and what is sent twice.

The instrument's media is archival — full-resolution stills because §4's masks
were cut from those pixels, one clip carrying both tracks because that is what
§3 plays. What a participant should be made to download is a different question,
and these are the answers to it: a web-sized copy where one can be made, the
source where one cannot, and in both cases enough of a cache to stop the same
bytes arriving a second time.

The two properties that matter to the study, rather than to the network, are
asserted here as well: the still keeps its dimensions, so a mark lands where it
was put, and the clip's audio is copied rather than re-encoded, so §5's mix is
the sound §3 played.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dpo.regen.app import build_app, warm_derivatives
from dpo.regen.derive import Derivatives
from dpo.regen.regeneration import RegenTemplateWriter

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to copy an audio track")


@pytest.fixture
def client(document: dict[str, Any], media_dir: Path, tmp_path: Path) -> TestClient:
    return TestClient(build_app(document, media_dir, tmp_path / "out", RegenTemplateWriter()))


@pytest.fixture
def undeceived(document: dict[str, Any], media_dir: Path, tmp_path: Path) -> TestClient:
    """The instrument with derivatives off: every response is the staged file."""
    return TestClient(build_app(document, media_dir, tmp_path / "out", RegenTemplateWriter(), derive=False))


def photograph(path: Path, size: tuple[int, int] = (320, 240)) -> None:
    """A still shaped like the ones the study stages: gradients under grain.

    The grain is the point. A clean synthetic gradient is PNG's best case and
    lossy WebP's worst — it encodes to a tenth of the WebP — so a test written
    on one would claim the opposite of what the staged footage does. Real
    frames are photographs of a street, where the filters PNG relies on are
    defeated by sensor noise and WebP is several times smaller. Seeded, so the
    sizes a test compares do not move between runs.
    """
    rng = random.Random(7)
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            base = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 3) % 256)
            image.putpixel((x, y), tuple(max(0, min(255, band + rng.randint(-12, 12))) for band in base))
    image.save(path)


class TestTheStill:
    def test_it_is_sent_as_webp_rather_than_as_the_staged_png(
        self, client: TestClient, media_dir: Path, document: dict[str, Any]
    ) -> None:
        source = media_dir / str(document["segments"]["A"]["frames"][0]["still"])
        photograph(source)
        response = client.get("/media/frame/A/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert len(response.content) < source.stat().st_size

    def test_it_keeps_the_dimensions_the_marks_are_normalised_against(
        self, client: TestClient, media_dir: Path, document: dict[str, Any]
    ) -> None:
        # §4 records a fraction of the delivered picture and matches it against
        # the staged mask. A resample here would move every mark by whatever
        # the two resolutions disagreed about.
        import io

        source = media_dir / str(document["segments"]["A"]["frames"][0]["still"])
        photograph(source, (321, 241))
        with Image.open(io.BytesIO(client.get("/media/frame/A/0").content)) as delivered:
            assert delivered.size == (321, 241)

    def test_a_still_that_cannot_be_encoded_is_served_as_it_was_staged(
        self, client: TestClient, media_dir: Path, document: dict[str, Any]
    ) -> None:
        # A session is not worth losing to an optimisation. The bytes on the
        # wire go up; nothing else changes.
        source = media_dir / str(document["segments"]["A"]["frames"][0]["still"])
        source.write_bytes(b"not an image at all")
        response = client.get("/media/frame/A/0")
        assert response.status_code == 200
        assert response.content == b"not an image at all"

    def test_derivatives_can_be_turned_off_entirely(
        self, undeceived: TestClient, media_dir: Path, document: dict[str, Any]
    ) -> None:
        source = media_dir / str(document["segments"]["A"]["frames"][0]["still"])
        photograph(source)
        response = undeceived.get("/media/frame/A/0")
        assert response.content == source.read_bytes()
        assert response.headers["content-type"] == "image/png"


class TestTheReferenceMix:
    @ffmpeg
    def test_it_is_the_sound_without_the_picture(self, media_dir: Path, tmp_path: Path) -> None:
        # Built here rather than in the fixture: the staged clip is a stub, and
        # this is the one test that needs a real container to copy out of.
        clip = media_dir / "A" / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(clip),
            ],
            check=True,
            capture_output=True,
        )
        derived, media_type = Derivatives(media_dir).sound(clip)
        assert derived != clip, "a clip with an audio track has a soundtrack to serve"
        assert media_type == "audio/mp4"
        assert derived.stat().st_size < clip.stat().st_size

    @ffmpeg
    def test_the_audio_is_copied_rather_than_re_encoded(self, media_dir: Path) -> None:
        """§5 compares a stem against the memory of §3. The mix has to be §3's sound.

        A re-encode would be a second, quieter rendering of it, and the
        comparison the screen asks for would be against something the
        participant never heard. So the bitstream is asserted, not the duration.
        """
        clip = media_dir / "A" / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(clip),
            ],
            check=True,
            capture_output=True,
        )

        def bitstream(path: Path) -> bytes:
            done = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-c:a", "copy", "-f", "md5", "-"],
                check=True,
                capture_output=True,
            )
            return done.stdout

        derived, _ = Derivatives(media_dir).sound(clip)
        assert bitstream(derived) == bitstream(clip)

    def test_a_clip_with_no_soundtrack_to_copy_is_served_whole(self, client: TestClient) -> None:
        # The fixture stages a stub for the clip, so nothing can be copied out
        # of it. §5 still gets a mix: the file the route always returned.
        response = client.get("/media/audio/A")
        assert response.status_code == 200
        assert response.content == b"\x00"

    def test_an_unknown_segment_has_no_mix(self, client: TestClient) -> None:
        assert client.get("/media/audio/Z").status_code == 404


class TestFetchingTwice:
    def test_a_still_the_browser_already_holds_is_not_sent_again(self, client: TestClient) -> None:
        first = client.get("/media/frame/A/0")
        assert first.status_code == 200
        again = client.get("/media/frame/A/0", headers={"if-none-match": first.headers["etag"]})
        assert again.status_code == 304
        assert again.content == b""

    def test_a_weak_tag_still_matches(self, client: TestClient) -> None:
        # The comparison a conditional GET calls for is the weak one, and a
        # proxy is entitled to weaken a tag on the way past.
        tag = client.get("/media/frame/A/0").headers["etag"]
        assert client.get("/media/frame/A/0", headers={"if-none-match": f"W/{tag}"}).status_code == 304

    def test_a_stale_tag_is_answered_with_the_picture(self, client: TestClient) -> None:
        response = client.get("/media/frame/A/0", headers={"if-none-match": '"from-another-staging"'})
        assert response.status_code == 200
        assert response.content

    def test_the_date_is_read_when_no_tag_is_offered(self, client: TestClient) -> None:
        first = client.get("/media/frame/A/0")
        again = client.get("/media/frame/A/0", headers={"if-modified-since": first.headers["last-modified"]})
        assert again.status_code == 304

    def test_media_says_how_long_it_may_be_kept(self, client: TestClient) -> None:
        # §4 is moved back and forth across. Without this the browser guesses,
        # and a guess of "ask again" is five more requests per crossing.
        assert "max-age" in client.get("/media/frame/A/0").headers["cache-control"]

    def test_the_page_revalidates_rather_than_being_kept(self, client: TestClient) -> None:
        # The opposite case: a fix to the instrument has to reach the next
        # reload, so the shell is stored and checked, never simply held.
        response = client.get("/regen.js")
        assert response.headers["cache-control"] == "no-cache"
        again = client.get("/regen.js", headers={"if-none-match": response.headers["etag"]})
        assert again.status_code == 304
        assert again.content == b""

    def test_an_edited_page_is_sent_again(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        held = client.get("/regen.js").headers["etag"]
        monkeypatch.setattr("dpo.regen.app._package_file", lambda name: "/* rebuilt */")
        response = client.get("/regen.js", headers={"if-none-match": held})
        assert response.status_code == 200
        assert response.text == "/* rebuilt */"


class TestWarming:
    def test_every_still_is_made_before_the_first_participant_asks(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        for name in ("A", "B"):
            for frame in document["segments"][name]["frames"]:
                photograph(media_dir / str(frame["still"]))
        stills, _, whole = warm_derivatives(document, media_dir)
        assert stills == 10, "five moments in each of two segments"
        assert whole == 2, "the fixture's clips are stubs with no audio to copy"

    def test_it_is_reported_rather_than_assumed(self, document: dict[str, Any], media_dir: Path) -> None:
        # A count short of what was asked for is the operator's signal that
        # something is being served whole, which is the thing worth knowing.
        (media_dir / str(document["segments"]["A"]["frames"][0]["still"])).write_bytes(b"broken")
        stills, _, whole = warm_derivatives(document, media_dir)
        assert stills == 9
        assert whole == 3


class TestTheCache:
    def test_a_restaged_still_is_not_served_from_the_old_one(self, media_dir: Path) -> None:
        """Re-staging replaces the file under a name the URL does not change.

        The key is the source's size and modification time, so the derivative
        of the old file cannot answer for the new one — which is what would
        otherwise put last week's footage in front of a participant.
        """
        source = media_dir / "A" / "frames" / "0.png"
        photograph(source, (320, 240))
        derivatives = Derivatives(media_dir)
        before, _ = derivatives.still(source)
        photograph(source, (321, 241))
        after, _ = derivatives.still(source)
        assert after != before

    def test_warming_and_serving_agree_on_what_a_still_is(
        self, media_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same file reached by two names is one derivative, not two.

        Warming walks the document from ``--media-dir`` as the operator typed
        it; the routes resolve each reference before serving. Were the key to
        follow the spelling, startup would fill a cache no request ever read.
        """
        source = media_dir / "A" / "frames" / "0.png"
        photograph(source)
        derivatives = Derivatives(media_dir)
        monkeypatch.chdir(media_dir.parent)
        by_relative, _ = derivatives.still(Path(source.relative_to(media_dir.parent).as_posix()))
        by_resolved, _ = derivatives.still(source.resolve())
        assert by_relative == by_resolved
        assert len(list((media_dir / ".derived").glob("*.webp"))) == 1

    def test_a_source_that_fails_is_not_retried_on_every_request(self, media_dir: Path) -> None:
        # Otherwise the first participant's failure becomes every
        # participant's, once per frame, for the length of the study.
        source = media_dir / "A" / "frames" / "0.png"
        source.write_bytes(b"broken")
        derivatives = Derivatives(media_dir)
        assert derivatives.still(source) == (source, None)
        attempts = 0
        original = Image.open

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Image, "open", counted)
            assert derivatives.still(source) == (source, None)
        assert attempts == 0

    def test_a_media_directory_it_cannot_write_to_costs_only_bandwidth(self, media_dir: Path) -> None:
        # A staging mounted read-only is a deployment, not a defect. The
        # instrument serves what it always served and says so once.
        source = media_dir / "A" / "frames" / "0.png"
        photograph(source)
        derivatives = Derivatives(media_dir)
        media_dir.chmod(0o555)
        try:
            assert derivatives.still(source) == (source, None)
        finally:
            media_dir.chmod(0o755)

    def test_nothing_is_written_beside_the_media_when_it_is_off(self, media_dir: Path) -> None:
        source = media_dir / "A" / "frames" / "0.png"
        photograph(source)
        assert Derivatives(media_dir, enabled=False).still(source) == (source, None)
        assert not (media_dir / ".derived").exists()
