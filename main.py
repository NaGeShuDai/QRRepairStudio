if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from qrrepair.selftest import run
        sys.exit(run(sys.argv[2]))
    from qrrepair.app import main
    main()
